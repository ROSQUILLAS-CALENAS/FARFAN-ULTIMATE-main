"""
Pattern Matcher Module

Implements pattern compilation and matching for query generation
with support for regex patterns and semantic similarity matching.
"""

import logging
import re
# # # from collections import defaultdict  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, List, Optional, Pattern, Set, Tuple, Union  # Module not found  # Module not found  # Module not found

import numpy as np


@dataclass
class PatternMatch:
    """Container for pattern match results"""

    pattern: str
    match_text: str
    start_pos: int
    end_pos: int
    confidence: float
    semantic_score: Optional[float] = None


@dataclass
class CompiledPattern:
    """Container for compiled pattern information"""

    original_pattern: str
    regex_pattern: Pattern
    tokens: List[str]
    pattern_type: str
    embedding: Optional[np.ndarray] = None


class PatternMatcher:
    """
    Advanced pattern matcher with regex and semantic matching capabilities.

    Supports compilation of pattern automata, exact matching, and semantic
    similarity-based matching for flexible query pattern recognition.
    """

    def __init__(
        self,
        semantic_model=None,
        similarity_threshold: float = 0.7,
        max_patterns: int = 1000,
        case_sensitive: bool = False,
    ):
        """
        Initialize pattern matcher.

        Args:
            semantic_model: Model for semantic similarity (e.g., SentenceTransformer)
            similarity_threshold: Minimum similarity for semantic matches
            max_patterns: Maximum number of patterns to store
            case_sensitive: Whether pattern matching is case sensitive
        """
        self.semantic_model = semantic_model
        self.similarity_threshold = similarity_threshold
        self.max_patterns = max_patterns
        self.case_sensitive = case_sensitive

        # Pattern storage
        self.compiled_patterns: Dict[str, CompiledPattern] = {}
        self.pattern_embeddings: Optional[np.ndarray] = None
        self.pattern_keys: List[str] = []

        # Pattern statistics
        self.match_statistics = defaultdict(int)

        self.logger = logging.getLogger(__name__)

    def compile_patterns(self, patterns: List[str]) -> Dict[str, CompiledPattern]:
        """
        Compile list of patterns into automata.

        Args:
            patterns: List of pattern strings

        Returns:
            Dictionary of compiled patterns
        """
        self.logger.info(f"Compiling {len(patterns)} patterns...")

        compiled = {}

        for pattern in patterns[: self.max_patterns]:
            compiled_pattern = self._compile_single_pattern(pattern)
            compiled[pattern] = compiled_pattern

        self.compiled_patterns.update(compiled)

        # Generate embeddings if semantic model available
        if self.semantic_model is not None:
            self._generate_pattern_embeddings()

        self.logger.info(f"Successfully compiled {len(compiled)} patterns")
        return compiled

    def _compile_single_pattern(self, pattern: str) -> CompiledPattern:
        """
        Compile a single pattern into automaton.

        Args:
            pattern: Pattern string

        Returns:
            Compiled pattern object
        """
        # Determine pattern type
        pattern_type = self._determine_pattern_type(pattern)

        # Create regex pattern
        regex_flags = 0 if self.case_sensitive else re.IGNORECASE

        if pattern_type == "literal":
            # Escape special regex characters for literal matching
            escaped_pattern = re.escape(pattern)
            regex_pattern = re.compile(escaped_pattern, regex_flags)
        elif pattern_type == "wildcard":
            # Convert wildcards to regex
            regex_str = pattern.replace("*", ".*").replace("?", ".")
            regex_pattern = re.compile(regex_str, regex_flags)
        elif pattern_type == "regex":
            # Use as-is for regex patterns
            regex_pattern = re.compile(pattern, regex_flags)
        else:
            # Default to literal
            escaped_pattern = re.escape(pattern)
            regex_pattern = re.compile(escaped_pattern, regex_flags)

        # Tokenize pattern
        tokens = self._tokenize_pattern(pattern)

        return CompiledPattern(
            original_pattern=pattern,
            regex_pattern=regex_pattern,
            tokens=tokens,
            pattern_type=pattern_type,
        )

    def _determine_pattern_type(self, pattern: str) -> str:
        """
        Determine the type of pattern (literal, wildcard, regex).

        Args:
            pattern: Pattern string

        Returns:
            Pattern type string
        """
        # Check for regex metacharacters
        regex_chars = set("[](){}^$+|\\")
        if any(char in pattern for char in regex_chars):
            return "regex"

        # Check for wildcards
        if "*" in pattern or "?" in pattern:
            return "wildcard"

        # Default to literal
        return "literal"

    def _tokenize_pattern(self, pattern: str) -> List[str]:
        """
        Tokenize pattern into constituent tokens.

        Args:
            pattern: Pattern string

        Returns:
            List of tokens
        """
        # Simple whitespace tokenization (can be enhanced)
        tokens = pattern.split()
        return tokens

    def _generate_pattern_embeddings(self):
        """Generate embeddings for compiled patterns using semantic model."""
        if not self.compiled_patterns or self.semantic_model is None:
            return

        patterns = list(self.compiled_patterns.keys())

        # Generate embeddings
        embeddings = self.semantic_model.encode(patterns, convert_to_numpy=True)

        # Store embeddings
        self.pattern_embeddings = embeddings
        self.pattern_keys = patterns

        # Update compiled patterns with embeddings
        for i, pattern in enumerate(patterns):
            self.compiled_patterns[pattern].embedding = embeddings[i]

        self.logger.info(f"Generated embeddings for {len(patterns)} patterns")

    def match_patterns(
        self, text: str, use_semantic: bool = True, max_matches: int = 10
    ) -> List[PatternMatch]:
        """
        Find pattern matches in text using exact and semantic matching.

        Args:
            text: Text to search
            use_semantic: Whether to use semantic similarity matching
            max_matches: Maximum number of matches to return

        Returns:
            List of pattern matches
        """
        matches = []

        # Exact pattern matching
        exact_matches = self._find_exact_matches(text)
        matches.extend(exact_matches)

        # Semantic matching if enabled and model available
        if (
            use_semantic
            and self.semantic_model is not None
            and self.pattern_embeddings is not None
        ):
            semantic_matches = self._find_semantic_matches(text)
            matches.extend(semantic_matches)

        # Remove duplicates and sort by confidence
        unique_matches = self._deduplicate_matches(matches)
        sorted_matches = sorted(
            unique_matches, key=lambda m: m.confidence, reverse=True
        )

        # Update statistics
        for match in sorted_matches[:max_matches]:
            self.match_statistics[match.pattern] += 1

        return sorted_matches[:max_matches]

    def _find_exact_matches(self, text: str) -> List[PatternMatch]:
        """
        Find exact pattern matches using compiled regex patterns.

        Args:
            text: Text to search

        Returns:
            List of exact matches
        """
        matches = []

        for pattern, compiled_pattern in self.compiled_patterns.items():
            # Find all matches for this pattern
            for match in compiled_pattern.regex_pattern.finditer(text):
                match_obj = PatternMatch(
                    pattern=pattern,
                    match_text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=1.0,  # Exact match has perfect confidence
                )
                matches.append(match_obj)

        return matches

    def _find_semantic_matches(self, text: str) -> List[PatternMatch]:
        """
        Find semantic similarity matches.

        Args:
            text: Text to search

        Returns:
            List of semantic matches
        """
        matches = []

        # Split text into sentences/phrases for semantic comparison
        phrases = self._extract_phrases(text)

        for phrase_text, start_pos, end_pos in phrases:
            # Get embedding for phrase
            phrase_embedding = self.semantic_model.encode(
                [phrase_text], convert_to_numpy=True
            )[0]

            # Compare with pattern embeddings
            similarities = np.dot(self.pattern_embeddings, phrase_embedding) / (
                np.linalg.norm(self.pattern_embeddings, axis=1)
                * np.linalg.norm(phrase_embedding)
            )

            # Find similar patterns above threshold
            similar_indices = np.where(similarities > self.similarity_threshold)[0]

            for idx in similar_indices:
                pattern = self.pattern_keys[idx]
                similarity_score = similarities[idx]

                match_obj = PatternMatch(
                    pattern=pattern,
                    match_text=phrase_text,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    confidence=similarity_score,
                    semantic_score=similarity_score,
                )
                matches.append(match_obj)

        return matches

    def _extract_phrases(self, text: str) -> List[Tuple[str, int, int]]:
        """
# # #         Extract phrases from text for semantic matching.  # Module not found  # Module not found  # Module not found

        Args:
            text: Input text

        Returns:
            List of (phrase, start_pos, end_pos) tuples
        """
        phrases = []

        # Split by sentences
        sentences = re.split(r"[.!?]+", text)
        current_pos = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Find sentence position in original text
            start_pos = text.find(sentence, current_pos)
            end_pos = start_pos + len(sentence)

            phrases.append((sentence, start_pos, end_pos))

# # #             # Also extract n-grams from sentence  # Module not found  # Module not found  # Module not found
            words = sentence.split()
            for n in [2, 3, 4]:  # bi-grams, tri-grams, 4-grams
                for i in range(len(words) - n + 1):
                    ngram = " ".join(words[i : i + n])
                    # Find n-gram position
                    ngram_start = text.find(ngram, start_pos)
                    if ngram_start != -1:
                        ngram_end = ngram_start + len(ngram)
                        phrases.append((ngram, ngram_start, ngram_end))

            current_pos = end_pos

        return phrases

    def _deduplicate_matches(self, matches: List[PatternMatch]) -> List[PatternMatch]:
        """
        Remove duplicate matches based on position and pattern.

        Args:
            matches: List of matches

        Returns:
            Deduplicated matches
        """
        seen = set()
        unique_matches = []

        for match in matches:
            # Create key based on pattern, start, and end position
            key = (match.pattern, match.start_pos, match.end_pos)

            if key not in seen:
                seen.add(key)
                unique_matches.append(match)
            else:
                # If we've seen this match, keep the one with higher confidence
                for i, existing_match in enumerate(unique_matches):
                    if (
                        existing_match.pattern == match.pattern
                        and existing_match.start_pos == match.start_pos
                        and existing_match.end_pos == match.end_pos
                    ):
                        if match.confidence > existing_match.confidence:
                            unique_matches[i] = match
                        break

        return unique_matches

    def find_pattern_facets(
        self, corpus_texts: List[str], min_facet_size: int = 5
    ) -> Dict[str, List[str]]:
        """
        Find corpus facets for each pattern based on matching frequency.

        Args:
            corpus_texts: List of corpus texts
            min_facet_size: Minimum number of matches to form a facet

        Returns:
            Dictionary mapping patterns to corpus facets
        """
        pattern_facets = defaultdict(list)

        for text_id, text in enumerate(corpus_texts):
            matches = self.match_patterns(text, use_semantic=True)

            for match in matches:
                pattern_facets[match.pattern].append(f"text_{text_id}")

        # Filter facets by minimum size
        filtered_facets = {}
        for pattern, facet_texts in pattern_facets.items():
            if len(facet_texts) >= min_facet_size:
                filtered_facets[pattern] = facet_texts

        self.logger.info(f"Found {len(filtered_facets)} pattern facets")
        return filtered_facets

    def get_pattern_statistics(self) -> Dict[str, int]:
        """Get pattern matching statistics."""
        return dict(self.match_statistics)

    def get_compiled_patterns(self) -> Dict[str, CompiledPattern]:
        """Get compiled patterns."""
        return self.compiled_patterns

    def add_patterns(self, new_patterns: List[str]) -> int:
        """
        Add new patterns to the matcher.

        Args:
            new_patterns: List of new pattern strings

        Returns:
            Number of patterns successfully added
        """
        if len(self.compiled_patterns) + len(new_patterns) > self.max_patterns:
            available_slots = self.max_patterns - len(self.compiled_patterns)
            new_patterns = new_patterns[:available_slots]
            self.logger.warning(
                f"Limited to {available_slots} new patterns due to max_patterns limit"
            )

        compiled_new = self.compile_patterns(new_patterns)
        return len(compiled_new)

    def remove_pattern(self, pattern: str) -> bool:
        """
# # #         Remove a pattern from the matcher.  # Module not found  # Module not found  # Module not found

        Args:
            pattern: Pattern to remove

        Returns:
            True if pattern was removed, False if not found
        """
        if pattern in self.compiled_patterns:
            del self.compiled_patterns[pattern]

            # Update embeddings if needed
            if self.semantic_model is not None:
                self._generate_pattern_embeddings()

            self.logger.info(f"Removed pattern: {pattern}")
            return True

        return False

    def clear_patterns(self):
        """Clear all compiled patterns."""
        self.compiled_patterns.clear()
        self.pattern_embeddings = None
        self.pattern_keys = []
        self.match_statistics.clear()
        self.logger.info("Cleared all patterns")

    def export_patterns(self) -> List[str]:
        """
        Export compiled patterns as list of strings.

        Returns:
            List of pattern strings
        """
        return list(self.compiled_patterns.keys())

    def get_pattern_embedding(self, pattern: str) -> Optional[np.ndarray]:
        """
        Get embedding for a specific pattern.

        Args:
            pattern: Pattern string

        Returns:
            Pattern embedding or None if not found
        """
        if pattern in self.compiled_patterns:
            return self.compiled_patterns[pattern].embedding
        return None

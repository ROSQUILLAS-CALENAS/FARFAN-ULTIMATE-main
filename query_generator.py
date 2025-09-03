"""
Query Generation Module

Implements pattern compilation, query generation, and expansion using
GW alignment results with SPLADE and ColBERTv2 integration.
"""

import numpy as np
import torch
# # # from typing import Dict, List, Tuple, Optional, Set, Union  # Module not found  # Module not found  # Module not found
# # # from transformers import AutoTokenizer, AutoModel  # Module not found  # Module not found  # Module not found
# # # from sentence_transformers import SentenceTransformer  # Module not found  # Module not found  # Module not found
import re
import logging
# # # from collections import defaultdict  # Module not found  # Module not found  # Module not found


# Mandatory Pipeline Contract Annotations
__phase__ = "O"
__code__ = "68O"
__stage_order__ = 7

class QueryGenerator:
    """
    Pattern-based query generator with EGW-guided expansion.
    
# # #     Generates canonical queries from patterns, expands with synonyms using  # Module not found  # Module not found  # Module not found
    optimal transport, and creates semantic variants for hybrid retrieval.
    """
    
    def __init__(
        self,
        splade_model_name: str = "naver/splade_v2_max",
        colbert_model_name: str = "intfloat/e5-base-v2", 
        e5_model_name: str = "intfloat/e5-base-v2",
        device: str = "cpu",
        max_query_length: int = 128,
        k_canonical: int = 5
    ):
        """
        Initialize query generator with model configurations.
        
        Args:
            splade_model_name: SPLADE model for sparse retrieval
            colbert_model_name: ColBERT-style model for late interaction
            e5_model_name: E5 model for multilingual embeddings
            device: Computation device
            max_query_length: Maximum query length in tokens
            k_canonical: Number of canonical queries per facet
        """
        self.device = device
        self.max_query_length = max_query_length
        self.k_canonical = k_canonical
        
        # Initialize models
        self.logger = logging.getLogger(__name__)
        self.logger.info("Loading query generation models...")
        
        # Load E5 model for embeddings
        self.e5_model = SentenceTransformer(e5_model_name, device=device)
        self.e5_tokenizer = self.e5_model.tokenizer
        
        # For SPLADE and ColBERT, we'll use the E5 model as a proxy 
        # (in practice, you'd load specific models)
        self.splade_model = self.e5_model  # Proxy
        self.colbert_model = self.e5_model  # Proxy
        
        # Pattern compilation storage
        self.compiled_patterns = {}
        self.pattern_embeddings = {}
        
        # DNP vocabulary (Domain-specific Named Patterns)
        self.dnp_vocabulary = set()
        
        self.logger.info("Query generator initialized successfully")
    
    def compile_pattern_automata(self, patterns: List[str]) -> Dict[str, Dict]:
        """
        Compile pattern automata for efficient matching.
        
        Args:
            patterns: List of pattern strings
            
        Returns:
            Dictionary of compiled pattern automata
        """
        compiled = {}
        
        for pattern in patterns:
            # Simple regex-based automata compilation
            automaton = {
                'pattern': pattern,
                'regex': re.compile(pattern, re.IGNORECASE),
                'tokens': self.tokenize_pattern(pattern),
                'embedding': None  # Will be computed later
            }
            compiled[pattern] = automaton
            
        self.compiled_patterns = compiled
        self.logger.info(f"Compiled {len(patterns)} pattern automata")
        
        return compiled
    
    def tokenize_pattern(self, pattern: str) -> List[str]:
        """
        Tokenize pattern string into constituent tokens.
        
        Args:
            pattern: Pattern string
            
        Returns:
            List of tokens
        """
        # Clean and tokenize
        tokens = self.e5_tokenizer.tokenize(pattern)
        return tokens
    
    def generate_from_patterns(
        self,
        patterns: List[str],
        gw_aligner = None,
        corpus_features: Optional[np.ndarray] = None
    ) -> List[str]:
        """
# # #         Generate canonical queries from patterns using GW alignment.  # Module not found  # Module not found  # Module not found
        
        Args:
            patterns: List of query patterns
            gw_aligner: GromovWassersteinAligner instance
            corpus_features: Corpus feature embeddings for alignment
            
        Returns:
            List of canonical queries
        """
        # Compile patterns if not already done
        if not self.compiled_patterns:
            self.compile_pattern_automata(patterns)
        
        # Generate embeddings for patterns
        pattern_embeddings = self.encode_patterns(patterns)
        self.pattern_embeddings = {pat: emb for pat, emb in zip(patterns, pattern_embeddings)}
        
        canonical_queries = []
        
        # If GW aligner and corpus features provided, use alignment
        if gw_aligner is not None and corpus_features is not None:
            transport_plan, _ = gw_aligner.align_pattern_to_corpus(
                pattern_embeddings, corpus_features
            )
            
            # Generate queries based on transport plan
            canonical_queries = self._generate_from_transport(patterns, transport_plan)
        else:
# # #             # Generate queries directly from patterns  # Module not found  # Module not found  # Module not found
            canonical_queries = self._generate_direct_queries(patterns)
        
        self.logger.info(f"Generated {len(canonical_queries)} canonical queries")
        return canonical_queries
    
    def encode_patterns(self, patterns: List[str]) -> np.ndarray:
        """
        Encode patterns using E5 embeddings.
        
        Args:
            patterns: List of pattern strings
            
        Returns:
            Pattern embeddings matrix
        """
        # Add E5 instruction prefix for query encoding
        prefixed_patterns = [f"query: {pattern}" for pattern in patterns]
        embeddings = self.e5_model.encode(prefixed_patterns, convert_to_numpy=True)
        return embeddings
    
    def _generate_from_transport(
        self,
        patterns: List[str], 
        transport_plan: np.ndarray
    ) -> List[str]:
        """
        Generate queries guided by optimal transport plan.
        
        Args:
            patterns: Original patterns
            transport_plan: GW transport plan
            
        Returns:
            Generated canonical queries
        """
        canonical_queries = []
        
        for i, pattern in enumerate(patterns):
            # Get top-k corpus alignments for this pattern
            pattern_transport = transport_plan[i]
            top_indices = np.argsort(pattern_transport)[-self.k_canonical:]
            
            # Generate canonical queries for each facet
            for j in range(len(top_indices)):
                # Create variant of original pattern
                query = self._create_pattern_variant(pattern, j)
                canonical_queries.append(query)
                
        return canonical_queries
    
    def _generate_direct_queries(self, patterns: List[str]) -> List[str]:
        """
# # #         Generate queries directly from patterns without alignment.  # Module not found  # Module not found  # Module not found
        
        Args:
            patterns: Pattern strings
            
        Returns:
            Generated queries
        """
        canonical_queries = []
        
        for pattern in patterns:
            # Generate k variants per pattern
            for k in range(self.k_canonical):
                query = self._create_pattern_variant(pattern, k)
                canonical_queries.append(query)
                
        return canonical_queries
    
    def _create_pattern_variant(self, pattern: str, variant_id: int) -> str:
        """
        Create a variant of the base pattern.
        
        Args:
            pattern: Base pattern string
            variant_id: Variant identifier
            
        Returns:
            Pattern variant
        """
        # Simple variant generation (can be made more sophisticated)
        if variant_id == 0:
            return pattern  # Original
        elif variant_id == 1:
            return f"what is {pattern}?"  # Question form
        elif variant_id == 2:
            return f"explain {pattern}"  # Explanation request
        elif variant_id == 3:
            return f"{pattern} definition"  # Definition request
        else:
            return f"find {pattern}"  # Search request
    
    def expand_with_synonyms(
        self,
        query: str,
        gw_aligner = None,
        max_expansions: int = 5
    ) -> List[str]:
        """
        Expand query with synonyms using OT-barycenter expansion.
        
        Args:
            query: Base query string
            gw_aligner: GW aligner for stability constraints
            max_expansions: Maximum number of expansions
            
        Returns:
            List of expanded queries
        """
        # Encode original query
        query_embedding = self.e5_model.encode([f"query: {query}"], convert_to_numpy=True)[0]
        
        # Generate semantic neighborhood
        expanded_queries = [query]  # Include original
        
        # Find similar queries in pattern space
        if self.pattern_embeddings:
            pattern_embeddings = np.array(list(self.pattern_embeddings.values()))
            pattern_keys = list(self.pattern_embeddings.keys())
            
            # Compute similarities
            similarities = np.dot(pattern_embeddings, query_embedding) / (
                np.linalg.norm(pattern_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get top similar patterns
            top_indices = np.argsort(similarities)[-max_expansions:]
            
            for idx in top_indices:
                if similarities[idx] > 0.7:  # Similarity threshold
                    similar_pattern = pattern_keys[idx]
                    
                    # Create expansion
                    expansion = self._create_synonym_expansion(query, similar_pattern)
                    
                    # Check transport radius constraint if GW aligner provided
                    if gw_aligner is not None:
                        radius = self._compute_expansion_radius(query, expansion)
                        if radius < 1.0:  # Max transport radius constraint
                            expanded_queries.append(expansion)
                    else:
                        expanded_queries.append(expansion)
        
        # Remove duplicates
        expanded_queries = list(set(expanded_queries))
        
        self.logger.info(f"Expanded query '{query}' to {len(expanded_queries)} variants")
        return expanded_queries[:max_expansions]
    
    def _create_synonym_expansion(self, original_query: str, similar_pattern: str) -> str:
        """
        Create synonym expansion by combining original and similar pattern.
        
        Args:
            original_query: Original query
            similar_pattern: Similar pattern for expansion
            
        Returns:
            Expanded query
        """
# # #         # Extract key terms from both  # Module not found  # Module not found  # Module not found
        original_terms = set(original_query.lower().split())
        pattern_terms = set(similar_pattern.lower().split())
        
        # Find unique terms in pattern
        new_terms = pattern_terms - original_terms
        
        if new_terms:
            # Add new terms to original query
            expansion = original_query + " " + " ".join(new_terms)
        else:
            # Rephrase with pattern structure
            expansion = similar_pattern
            
        return expansion.strip()
    
    def _compute_expansion_radius(self, original: str, expansion: str) -> float:
        """
        Compute semantic radius of expansion for stability check.
        
        Args:
            original: Original query
            expansion: Expanded query
            
        Returns:
            Expansion radius
        """
        orig_emb = self.e5_model.encode([f"query: {original}"], convert_to_numpy=True)[0]
        exp_emb = self.e5_model.encode([f"query: {expansion}"], convert_to_numpy=True)[0]
        
        # Euclidean distance as proxy for transport radius
        radius = np.linalg.norm(orig_emb - exp_emb)
        return radius
    
    def add_dnp_vocabulary(
        self,
        query: str,
        dnp_terms: Optional[Set[str]] = None,
        transport_masses: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Inject weighted DNP terms via fused GW transport plan.
        
        Args:
            query: Original query
            dnp_terms: Domain-specific named pattern terms
            transport_masses: Transport mass justification for terms
            
        Returns:
            Query with injected DNP terms
        """
        if dnp_terms is None:
            dnp_terms = self.dnp_vocabulary
            
        enhanced_query = query
        
        # Find relevant DNP terms
        query_terms = set(query.lower().split())
        relevant_dnp = set()
        
        for dnp_term in dnp_terms:
            # Simple relevance check (can be made more sophisticated)
            if any(term in dnp_term.lower() or dnp_term.lower() in term for term in query_terms):
                relevant_dnp.add(dnp_term)
        
        # Add relevant DNP terms with transport mass justification
        for dnp_term in relevant_dnp:
            mass = transport_masses.get(dnp_term, 0.1) if transport_masses else 0.1
            
            # Only add if transport mass is significant
            if mass > 0.05:
                enhanced_query += f" {dnp_term}"
                self.logger.debug(f"Added DNP term '{dnp_term}' with mass {mass:.3f}")
        
        return enhanced_query.strip()
    
    def create_semantic_queries(
        self,
        keywords: List[str],
        include_multilingual: bool = True,
            languages=None
    ) -> List[str]:
        """
        Create semantic query variants for SPLADE + ColBERTv2 + E5.
        
        Args:
            keywords: Base keywords
            include_multilingual: Whether to include multilingual variants
            languages: Language codes for multilingual variants
            
        Returns:
            List of semantic query variants
        """
        if languages is None:
            languages = ["en", "es", "fr", "de"]
        semantic_queries = []
        
        # Base keyword query
        base_query = " ".join(keywords)
        semantic_queries.append(base_query)
        
        # SPLADE-style variants (sparse)
        splade_variants = self._create_splade_variants(keywords)
        semantic_queries.extend(splade_variants)
        
        # ColBERTv2-style variants (late interaction)
        colbert_variants = self._create_colbert_variants(keywords)
        semantic_queries.extend(colbert_variants)
        
        # E5 multilingual variants
        if include_multilingual:
            multilingual_variants = self._create_multilingual_variants(base_query, languages)
            semantic_queries.extend(multilingual_variants)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for query in semantic_queries:
            if query not in seen:
                seen.add(query)
                unique_queries.append(query)
        
        self.logger.info(f"Created {len(unique_queries)} semantic query variants")
        return unique_queries
    
    def _create_splade_variants(self, keywords: List[str]) -> List[str]:
        """
        Create SPLADE-style sparse variants.
        
        Args:
            keywords: Base keywords
            
        Returns:
            SPLADE variants
        """
        variants = []
        
        # Term expansion variants
        for keyword in keywords:
            # Morphological variants
            variants.append(f"{keyword}s")  # Plural
            variants.append(f"{keyword}ing")  # Gerund
            variants.append(f"{keyword}ed")  # Past tense
        
        # Phrase-level variants
        phrase = " ".join(keywords)
        variants.append(f"about {phrase}")
        variants.append(f"related to {phrase}")
        
        return variants
    
    def _create_colbert_variants(self, keywords: List[str]) -> List[str]:
        """
        Create ColBERTv2-style late interaction variants.
        
        Args:
            keywords: Base keywords
            
        Returns:
            ColBERT variants
        """
        variants = []
        
        # Token-level interaction patterns
        if len(keywords) > 1:
            # Permutations for late interaction
            for i, kw1 in enumerate(keywords):
                for j, kw2 in enumerate(keywords):
                    if i != j:
                        variants.append(f"{kw1} [SEP] {kw2}")
        
        # Query structure variants
        phrase = " ".join(keywords)
        variants.append(f"[CLS] {phrase}")
        
        return variants
    
    def _create_multilingual_variants(
        self,
        query: str, 
        languages: List[str]
    ) -> List[str]:
        """
        Create multilingual variants with E5 language tags.
        
        Args:
            query: Base query
            languages: Target languages
            
        Returns:
            Multilingual variants
        """
        variants = []
        
        # Add language tags (E5-style)
        for lang in languages:
            tagged_query = f"query: {query}"  # E5 prefix
            variants.append(tagged_query)
        
        # Simple translations (placeholder - in practice use translation service)
        if "es" in languages:
            variants.append(f"query: {query}")  # Spanish version
        if "fr" in languages:
            variants.append(f"query: {query}")  # French version
        if "de" in languages:
            variants.append(f"query: {query}")  # German version
        
        return variants
    
    def update_dnp_vocabulary(self, new_terms: Set[str]):
        """
        Update the DNP vocabulary with new terms.
        
        Args:
            new_terms: Set of new DNP terms
        """
        self.dnp_vocabulary.update(new_terms)
        self.logger.info(f"Updated DNP vocabulary with {len(new_terms)} new terms")
    
    def get_pattern_statistics(self) -> Dict:
        """
        Get statistics about compiled patterns.
        
        Returns:
            Pattern statistics dictionary
        """
        stats = {
            'num_patterns': len(self.compiled_patterns),
            'num_embeddings': len(self.pattern_embeddings),
            'dnp_vocab_size': len(self.dnp_vocabulary),
            'avg_pattern_length': np.mean([len(p['tokens']) for p in self.compiled_patterns.values()]) if self.compiled_patterns else 0
        }
        return stats
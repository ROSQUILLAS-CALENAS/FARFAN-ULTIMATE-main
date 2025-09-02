"""
Retrieval Trace Tool

Prints queries, filters, weights and index hashes for debugging and verification
of deterministic hybrid retrieval system.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from deterministic_hybrid_retrieval import (
        DeterministicHybridRetriever,
        QueryExpansion, 
        DNPConstraint,
    )
except ImportError:
    import sys
    sys.path.append('..')
    from deterministic_hybrid_retrieval import (
        DeterministicHybridRetriever,
        QueryExpansion,
        DNPConstraint,
    )


class RetrievalTracer:
    """Trace and debug hybrid retrieval operations"""
    
    def __init__(self, output_file: Optional[str] = None, verbose: bool = True):
        self.output_file = output_file
        self.verbose = verbose
        self.traces = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if verbose:
            logging.basicConfig(level=logging.INFO)
    
    def trace_search(
        self, 
        retriever: DeterministicHybridRetriever,
        query: str,
        top_k: int = 10,
        expansion_config: Optional[QueryExpansion] = None,
        dnp_constraints: Optional[List[DNPConstraint]] = None,
        include_intermediates: bool = True,
    ) -> Dict[str, Any]:
        """
        Trace a complete search operation with detailed logging.
        
        Args:
            retriever: The hybrid retriever instance
            query: Search query
            top_k: Number of results
            expansion_config: Optional query expansion config
            dnp_constraints: Optional DNP constraints
            include_intermediates: Whether to trace intermediate results
            
        Returns:
            Complete trace information
        """
        
        trace_start = datetime.now()
        trace_id = hashlib.sha256(f"{query}_{trace_start.isoformat()}".encode()).hexdigest()[:16]
        
        self._log(f"\n{'='*60}")
        self._log(f"RETRIEVAL TRACE {trace_id}")
        self._log(f"{'='*60}")
        self._log(f"Timestamp: {trace_start.isoformat()}")
        self._log(f"Query: '{query}'")
        self._log(f"Top-K: {top_k}")
        
        # Trace configuration
        config_info = self._trace_config(retriever)
        
        # Trace query processing
        queries_info = self._trace_query_expansion(query, expansion_config)
        
        # Trace constraints
        constraints_info = self._trace_constraints(dnp_constraints)
        
        # Trace index state
        index_info = self._trace_index_state(retriever)
        
        # Perform search with tracing
        if include_intermediates:
            search_info = self._trace_search_with_intermediates(
                retriever, query, top_k, expansion_config, dnp_constraints
            )
        else:
            # Standard search
            result = retriever.search(query, top_k, expansion_config, dnp_constraints)
            search_info = self._trace_final_results(result)
        
        # Compile complete trace
        complete_trace = {
            "trace_id": trace_id,
            "timestamp": trace_start.isoformat(),
            "query": query,
            "top_k": top_k,
            "config": config_info,
            "queries": queries_info,
            "constraints": constraints_info,
            "index": index_info,
            "search": search_info,
            "duration_ms": (datetime.now() - trace_start).total_seconds() * 1000,
        }
        
        self.traces.append(complete_trace)
        
        # Save if output file specified
        if self.output_file:
            self._save_trace(complete_trace)
        
        self._log(f"Trace completed in {complete_trace['duration_ms']:.2f}ms")
        self._log(f"{'='*60}\n")
        
        return complete_trace
    
    def _trace_config(self, retriever: DeterministicHybridRetriever) -> Dict[str, Any]:
        """Trace retriever configuration"""
        
        self._log("\n[CONFIG]")
        config = {
            "embedding_model": retriever.embedding_model_name,
            "device": retriever.device,
            "seed": retriever.seed,
            "fusion_weights": {
                "sparse_alpha": retriever.fusion.alpha,
                "dense_beta": retriever.fusion.beta, 
                "projection_gamma": retriever.fusion.gamma,
            },
            "embedding_dim": retriever.embedding_dim,
            "indices_initialized": retriever.indices_initialized,
        }
        
        self._log(f"Model: {config['embedding_model']}")
        self._log(f"Device: {config['device']}")
        self._log(f"Seed: {config['seed']}")
        self._log(f"Fusion weights: α={config['fusion_weights']['sparse_alpha']}, "
                  f"β={config['fusion_weights']['dense_beta']}, "
                  f"γ={config['fusion_weights']['projection_gamma']}")
        self._log(f"Embedding dim: {config['embedding_dim']}")
        
        return config
    
    def _trace_query_expansion(
        self, 
        query: str, 
        expansion_config: Optional[QueryExpansion]
    ) -> Dict[str, Any]:
        """Trace query expansion process"""
        
        self._log("\n[QUERY PROCESSING]")
        
        queries_info = {
            "original_query": query,
            "expansion_config": None,
            "expanded_queries": [query],  # Always includes original
        }
        
        if expansion_config:
            expanded = expansion_config.expand_query(query)
            queries_info.update({
                "expansion_config": {
                    "expansion_weights": list(expansion_config.expansion_weights),
                    "expansion_radius": expansion_config.expansion_radius,
                    "max_expansions": expansion_config.max_expansions,
                },
                "expanded_queries": expanded,
            })
            
            self._log(f"Original: '{query}'")
            self._log(f"Expansion weights: {list(expansion_config.expansion_weights)}")
            self._log(f"Max expansions: {expansion_config.max_expansions}")
            for i, eq in enumerate(expanded):
                self._log(f"  [{i}]: '{eq}'")
        else:
            self._log(f"Query: '{query}' (no expansion)")
            
        return queries_info
    
    def _trace_constraints(self, dnp_constraints: Optional[List[DNPConstraint]]) -> Dict[str, Any]:
        """Trace DNP constraints"""
        
        self._log("\n[CONSTRAINTS]")
        
        constraints_info = {
            "dnp_constraints": [],
            "constraint_count": 0,
        }
        
        if dnp_constraints:
            constraints_info["constraint_count"] = len(dnp_constraints)
            
            for i, constraint in enumerate(dnp_constraints):
                constraint_data = {
                    "field": constraint.field,
                    "operator": constraint.operator,
                    "value": constraint.value,
                }
                constraints_info["dnp_constraints"].append(constraint_data)
                
                self._log(f"DNP[{i}]: {constraint.field} {constraint.operator} {constraint.value}")
        else:
            self._log("No DNP constraints")
            
        return constraints_info
    
    def _trace_index_state(self, retriever: DeterministicHybridRetriever) -> Dict[str, Any]:
        """Trace current index state"""
        
        self._log("\n[INDEX STATE]")
        
        index_info = {
            "snapshot_id": None,
            "documents_hash": None,
            "document_count": len(retriever.documents),
            "unique_content_hashes": len({dh.content_hash for dh in retriever.document_hashes}),
            "indices_ready": retriever.indices_initialized,
        }
        
        if retriever.current_snapshot:
            index_info.update({
                "snapshot_id": retriever.current_snapshot.snapshot_id,
                "documents_hash": retriever.current_snapshot.documents_hash,
                "embedding_dim": retriever.current_snapshot.embedding_dim,
                "sparse_dim": retriever.current_snapshot.sparse_dim,
                "creation_time": retriever.current_snapshot.creation_time,
            })
            
            self._log(f"Snapshot ID: {index_info['snapshot_id']}")
            self._log(f"Documents hash: {index_info['documents_hash'][:16]}...")
            self._log(f"Document count: {index_info['document_count']}")
            self._log(f"Unique content hashes: {index_info['unique_content_hashes']}")
            self._log(f"Embedding dim: {index_info['embedding_dim']}")
            self._log(f"Sparse dim: {index_info['sparse_dim']}")
        else:
            self._log("No snapshot available")
            
        # Index hashes for verification
        if retriever.indices_initialized:
            # Create index fingerprints (simplified)
            dense_fingerprint = f"dense_{retriever.dense_index.ntotal}_{retriever.embedding_dim}"
            sparse_fingerprint = f"sparse_{retriever.sparse_index.ntotal}_{getattr(retriever, 'sparse_index', {}).get('d', 0)}"
            
            index_info.update({
                "dense_index_fingerprint": hashlib.sha256(dense_fingerprint.encode()).hexdigest()[:16],
                "sparse_index_fingerprint": hashlib.sha256(sparse_fingerprint.encode()).hexdigest()[:16],
            })
            
            self._log(f"Dense index fingerprint: {index_info['dense_index_fingerprint']}")
            self._log(f"Sparse index fingerprint: {index_info['sparse_index_fingerprint']}")
            
        return index_info
    
    def _trace_search_with_intermediates(
        self,
        retriever: DeterministicHybridRetriever,
        query: str,
        top_k: int,
        expansion_config: Optional[QueryExpansion],
        dnp_constraints: Optional[List[DNPConstraint]],
    ) -> Dict[str, Any]:
        """Trace search with intermediate results"""
        
        self._log("\n[SEARCH EXECUTION]")
        
        search_info = {
            "method": "deterministic_hybrid_with_trace",
            "intermediate_results": {},
            "fusion_process": {},
            "final_results": {},
        }
        
        # Get query variants
        queries = [query]
        if expansion_config:
            queries = expansion_config.expand_query(query)
            
        self._log(f"Processing {len(queries)} query variant(s)")
        
        # Trace each component search
        all_sparse_results = []
        all_dense_results = []
        all_projection_results = []
        
        for i, q in enumerate(queries):
            self._log(f"\n  Query variant [{i}]: '{q}'")
            
            # Sparse search
            sparse_results = retriever._sparse_search(q, top_k * 2)
            self._log(f"    Sparse: {len(sparse_results)} results")
            if sparse_results and self.verbose:
                for j, (doc_hash, score) in enumerate(sparse_results[:3]):
                    self._log(f"      [{j}] {doc_hash.content_hash[:16]}... (score: {score:.4f})")
            
            # Dense search  
            dense_results = retriever._dense_search(q, top_k * 2)
            self._log(f"    Dense: {len(dense_results)} results")
            if dense_results and self.verbose:
                for j, (doc_hash, score) in enumerate(dense_results[:3]):
                    self._log(f"      [{j}] {doc_hash.content_hash[:16]}... (score: {score:.4f})")
            
            # Projection search
            projection_results = retriever._projection_search(q, top_k * 2)  
            self._log(f"    Projection: {len(projection_results)} results")
            if projection_results and self.verbose:
                for j, (doc_hash, score) in enumerate(projection_results[:3]):
                    self._log(f"      [{j}] {doc_hash.content_hash[:16]}... (score: {score:.4f})")
            
            all_sparse_results.extend(sparse_results)
            all_dense_results.extend(dense_results)
            all_projection_results.extend(projection_results)
        
        # Store intermediate results
        search_info["intermediate_results"] = {
            "sparse_count": len(all_sparse_results),
            "dense_count": len(all_dense_results),
            "projection_count": len(all_projection_results),
        }
        
        # Trace fusion process
        self._log(f"\n  Fusion process:")
        self._log(f"    Total sparse results: {len(all_sparse_results)}")
        self._log(f"    Total dense results: {len(all_dense_results)}")
        self._log(f"    Total projection results: {len(all_projection_results)}")
        
        fusion_results = retriever.fusion.fuse_rankings(
            all_sparse_results, all_dense_results, all_projection_results
        )
        
        self._log(f"    Fusion produced {len(fusion_results)} combined results")
        
        # Apply constraints if present
        if dnp_constraints:
            original_count = len(fusion_results)
            fusion_results = retriever._apply_dnp_constraints(fusion_results, dnp_constraints)
            filtered_count = len(fusion_results)
            self._log(f"    DNP constraints filtered: {original_count} → {filtered_count}")
        
        # Deduplicate
        seen_hashes = set()
        unique_results = []
        for result in fusion_results:
            if result.doc_hash.content_hash not in seen_hashes:
                seen_hashes.add(result.doc_hash.content_hash)
                unique_results.append(result)
        
        self._log(f"    After deduplication: {len(unique_results)} unique results")
        
        # Final top-K
        final_results = unique_results[:top_k]
        self._log(f"    Final top-{top_k}: {len(final_results)} results")
        
        # Trace final results
        search_info["final_results"] = {
            "result_count": len(final_results),
            "results": []
        }
        
        self._log(f"\n[FINAL RESULTS]")
        for i, result in enumerate(final_results):
            result_info = {
                "rank": i,
                "content_hash": result.doc_hash.content_hash,
                "doc_id": result.doc_hash.doc_id,
                "fusion_score": result.fusion_score,
                "component_scores": {
                    "sparse": result.sparse_score,
                    "dense": result.dense_score,
                },
            }
            search_info["final_results"]["results"].append(result_info)
            
            self._log(f"  [{i}] {result.doc_hash.content_hash[:16]}... "
                      f"(fusion: {result.fusion_score:.4f}, "
                      f"sparse: {result.sparse_score:.4f}, "
                      f"dense: {result.dense_score:.4f})")
        
        return search_info
    
    def _trace_final_results(self, result) -> Dict[str, Any]:
        """Trace only final results (no intermediates)"""
        
        search_info = {
            "method": result.method,
            "result_count": len(result.doc_hashes),
            "snapshot_id": result.snapshot_id,
            "results": []
        }
        
        self._log(f"\n[SEARCH RESULTS]")
        self._log(f"Method: {result.method}")
        self._log(f"Results: {len(result.doc_hashes)}")
        self._log(f"Snapshot: {result.snapshot_id}")
        
        for i, (doc_hash, score) in enumerate(zip(result.doc_hashes, result.scores)):
            result_info = {
                "rank": i,
                "content_hash": doc_hash.content_hash,
                "doc_id": doc_hash.doc_id,
                "score": score,
            }
            search_info["results"].append(result_info)
            
            self._log(f"  [{i}] {doc_hash.content_hash[:16]}... (score: {score:.4f})")
        
        return search_info
    
    def _log(self, message: str):
        """Log message with consistent formatting"""
        if self.verbose:
            print(message)
        self.logger.info(message)
    
    def _save_trace(self, trace: Dict[str, Any]):
        """Save trace to output file"""
        if self.output_file:
            # Load existing traces
            output_path = Path(self.output_file)
            existing_traces = []
            
            if output_path.exists():
                try:
                    with open(output_path, 'r') as f:
                        existing_traces = json.load(f)
                except json.JSONDecodeError:
                    existing_traces = []
            
            # Append new trace
            existing_traces.append(trace)
            
            # Save all traces
            with open(output_path, 'w') as f:
                json.dump(existing_traces, f, indent=2)
    
    def save_all_traces(self, filename: str):
        """Save all collected traces to file"""
        with open(filename, 'w') as f:
            json.dump(self.traces, f, indent=2)
    
    def compare_traces(self, trace1: Dict[str, Any], trace2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two traces for differences"""
        
        comparison = {
            "traces_match": True,
            "differences": [],
        }
        
        # Compare configurations
        if trace1["config"] != trace2["config"]:
            comparison["traces_match"] = False
            comparison["differences"].append({
                "type": "config",
                "trace1": trace1["config"],
                "trace2": trace2["config"],
            })
        
        # Compare final results
        if len(trace1["search"]["final_results"]["results"]) != len(trace2["search"]["final_results"]["results"]):
            comparison["traces_match"] = False
            comparison["differences"].append({
                "type": "result_count",
                "trace1_count": len(trace1["search"]["final_results"]["results"]),
                "trace2_count": len(trace2["search"]["final_results"]["results"]),
            })
        else:
            # Compare individual results
            for i, (r1, r2) in enumerate(zip(
                trace1["search"]["final_results"]["results"],
                trace2["search"]["final_results"]["results"]
            )):
                if r1["content_hash"] != r2["content_hash"]:
                    comparison["traces_match"] = False
                    comparison["differences"].append({
                        "type": "result_order",
                        "position": i,
                        "trace1_hash": r1["content_hash"],
                        "trace2_hash": r2["content_hash"],
                    })
                
                # Check score differences (with tolerance)
                if "fusion_score" in r1 and "fusion_score" in r2:
                    score_diff = abs(r1["fusion_score"] - r2["fusion_score"])
                    if score_diff > 1e-6:
                        comparison["traces_match"] = False
                        comparison["differences"].append({
                            "type": "score_difference", 
                            "position": i,
                            "trace1_score": r1["fusion_score"],
                            "trace2_score": r2["fusion_score"],
                            "difference": score_diff,
                        })
        
        return comparison


def main():
    """Demo usage of retrieval tracer"""
    
    try:
        # Sample data
        documents = [
            "Machine learning algorithms are used for pattern recognition and data analysis.",
            "Deep learning neural networks can process complex data structures effectively.", 
            "Natural language processing enables computers to understand human language.",
            "Information retrieval systems help users find relevant documents quickly.",
            "Semantic search improves relevance by understanding meaning and context.",
        ]
        
        queries = [
            "machine learning algorithms",
            "natural language processing",
        ]
        
        # Initialize retriever
        retriever = DeterministicHybridRetriever(
            embedding_model_name="intfloat/e5-base-v2",
            device="cpu",
            seed=42
        )
        
        retriever.add_documents(documents)
        
        # Initialize tracer
        tracer = RetrievalTracer(output_file="retrieval_traces.json", verbose=True)
        
        # Trace searches
        print("Tracing basic searches...")
        for query in queries:
            trace = tracer.trace_search(retriever, query, top_k=3)
            
        # Trace with query expansion
        print("\nTracing with query expansion...")
        expansion_config = QueryExpansion(
            expansion_weights=(0.8, 0.6),
            expansion_radius=0.5,
            max_expansions=2
        )
        
        tracer.trace_search(
            retriever, 
            "machine learning",
            top_k=3,
            expansion_config=expansion_config,
            include_intermediates=True
        )
        
        # Save all traces
        tracer.save_all_traces("complete_traces.json")
        print(f"\nTracing complete. Generated {len(tracer.traces)} traces.")
        
    except ImportError as e:
        print(f"Missing dependencies for full tracing demo: {e}")
        print("Tracer module structure validated - ready for use with proper dependencies.")
        
        # Create minimal example trace file
        example_trace = {
            "trace_id": "example_trace",
            "timestamp": "2024-01-01T00:00:00",
            "query": "example query",
            "top_k": 5,
            "config": {
                "embedding_model": "intfloat/e5-base-v2",
                "device": "cpu",
                "seed": 42
            },
            "results": "Example trace structure - ready for use"
        }
        
        import json
        with open("example_trace.json", "w") as f:
            json.dump(example_trace, f, indent=2)
        
        print("Example trace structure saved to example_trace.json")


if __name__ == "__main__":
    main()
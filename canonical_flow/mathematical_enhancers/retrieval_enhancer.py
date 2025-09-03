"""
Mathematical Stage 6: Spectral Graph Theory Retrieval Enhancer

This module applies spectral graph theory and eigenvalue decomposition to enhance
the hybrid retrieval system's optimization and semantic reranking capabilities.

Key Features:
1. Document Similarity Graph Construction
2. Eigenvalue Decomposition and Spectral Analysis  
3. Spectral Clustering for Semantic Reranking
4. Mathematical Validation and Stability Bounds
5. Integration with Existing Retrieval Components

Theoretical Foundation:
- Spectral graph theory for document clustering and similarity analysis
- Eigenvalue stability bounds via Weyl's inequalities and matrix perturbation theory
- Spectral clustering via normalized Laplacian eigenvectors
- Numerical stability through condition number monitoring and regularization
"""

# # # from __future__ import annotations  # Module not found  # Module not found  # Module not found

import math
import os
import warnings
# # # from collections import defaultdict  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Tuple, Union  # Module not found  # Module not found  # Module not found

import numpy as np
# # # from scipy import linalg  # Module not found  # Module not found  # Module not found
# # # from scipy.sparse import csr_matrix, diags  # Module not found  # Module not found  # Module not found
# # # from scipy.sparse.linalg import eigsh, ArpackError  # Module not found  # Module not found  # Module not found
# # # from sklearn.cluster import SpectralClustering  # Module not found  # Module not found  # Module not found
# # # from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel  # Module not found  # Module not found  # Module not found
# # # from sklearn.preprocessing import normalize  # Module not found  # Module not found  # Module not found

# # # from .hyperbolic_tensor_networks import HyperbolicTensorNetworks  # Module not found  # Module not found  # Module not found

try:
    pass  # Added to fix syntax
# # # #     from orchestration.event_bus import publish_metric  # type: ignore  # Module not found  # Module not found  # Module not found  # Module not found
except Exception:  # noqa: BLE001
    def publish_metric(topic: str, payload: Dict[str, Any]) -> None:  # type: ignore
        return None

try:
    pass  # Added to fix syntax
# # # #     from tracing.decorators import trace  # type: ignore  # Module not found  # Module not found  # Module not found  # Module not found
except Exception:  # noqa: BLE001
    def trace(fn):  # type: ignore
        return fn


@trace
def _compute_document_similarity_graph(embeddings: np.ndarray, 
                                      method: str = "cosine",
                                      sigma: float = 1.0,
                                      k_nearest: int = 10) -> np.ndarray:
    """
# # #     Construct document similarity graph from embeddings.  # Module not found  # Module not found  # Module not found
    
    Args:
        embeddings: Document embeddings (n_docs, dim)
        method: Similarity method ("cosine", "rbf", "linear")  
        sigma: RBF kernel parameter
        k_nearest: Number of nearest neighbors to keep
        
    Returns:
        Similarity matrix (n_docs, n_docs)
    """
    if embeddings.size == 0:
        return np.zeros((0, 0))
    
    n_docs = embeddings.shape[0]
    
    if method == "cosine":
        # Cosine similarity
        similarity = cosine_similarity(embeddings)
        
    elif method == "rbf":
        # RBF (Gaussian) kernel  
        similarity = rbf_kernel(embeddings, gamma=1.0 / (2 * sigma**2))
        
    elif method == "linear":
        # Linear kernel (dot product)
        similarity = np.dot(embeddings, embeddings.T)
        # Normalize to [0,1]
        similarity = (similarity - similarity.min()) / (similarity.max() - similarity.min() + 1e-10)
        
    else:
        raise ValueError(f"Unknown similarity method: {method}")
    
    # Convert to k-nearest neighbors graph for sparsity
    if k_nearest > 0 and k_nearest < n_docs:
        # Keep only k largest similarities per row
        for i in range(n_docs):
            row = similarity[i]
            top_k_indices = np.argsort(row)[-k_nearest:]
            mask = np.zeros(n_docs, dtype=bool)
            mask[top_k_indices] = True
            similarity[i, ~mask] = 0
    
    # Ensure symmetry
    similarity = (similarity + similarity.T) / 2
    
    # Remove self-loops for cleaner spectral properties
    np.fill_diagonal(similarity, 0)
    
    return similarity


@trace  
def _compute_graph_laplacian(similarity_matrix: np.ndarray, 
                            normalized: bool = True,
                            regularization: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
# # #     Compute graph Laplacian matrix from similarity matrix.  # Module not found  # Module not found  # Module not found
    
    Args:
        similarity_matrix: Symmetric similarity matrix
        normalized: Whether to compute normalized Laplacian
        regularization: Regularization parameter for numerical stability
        
    Returns:
        (laplacian_matrix, degree_vector)
    """
    if similarity_matrix.size == 0:
        return np.zeros((0, 0)), np.zeros(0)
    
    # Degree matrix
    degree = np.sum(similarity_matrix, axis=1)
    
    # Add regularization to avoid zero degrees
    degree = degree + regularization
    
    if normalized:
        # Normalized Laplacian: L_norm = I - D^(-1/2) W D^(-1/2)
        degree_sqrt_inv = np.where(degree > 0, 1.0 / np.sqrt(degree), 0)
        degree_matrix_sqrt_inv = diags(degree_sqrt_inv)
        
        if similarity_matrix.shape[0] > 1000:
            # Use sparse matrices for large graphs
            similarity_sparse = csr_matrix(similarity_matrix)
            laplacian = diags(np.ones(len(degree))) - degree_matrix_sqrt_inv @ similarity_sparse @ degree_matrix_sqrt_inv
            laplacian = laplacian.toarray()
        else:
            D_sqrt_inv = np.diag(degree_sqrt_inv)
            laplacian = np.eye(len(degree)) - D_sqrt_inv @ similarity_matrix @ D_sqrt_inv
    else:
        # Unnormalized Laplacian: L = D - W
        laplacian = np.diag(degree) - similarity_matrix
    
    return laplacian, degree


@trace
def _compute_eigendecomposition(matrix: np.ndarray, 
                               k: int = 10,
                               which: str = 'SM') -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Compute eigendecomposition with stability analysis.
    
    Args:
        matrix: Symmetric matrix to decompose
        k: Number of eigenvalues/eigenvectors to compute
        which: Which eigenvalues to find ('SM' for smallest, 'LM' for largest)
        
    Returns:
        (eigenvalues, eigenvectors, stability_metrics)
    """
    if matrix.size == 0:
        return np.zeros(0), np.zeros((0, 0)), {"condition_number": 1.0, "spectral_gap": 0.0}
    
    n = matrix.shape[0]
    k = min(k, n - 1)
    
    try:
        # Use sparse eigenvalue solver for efficiency
        if n > 100 and which in ['SM', 'LM']:
            eigenvals, eigenvecs = eigsh(matrix, k=k, which=which, tol=1e-10, maxiter=1000)
        else:
            # Full eigendecomposition for smaller matrices
            eigenvals, eigenvecs = linalg.eigh(matrix)
            if which == 'SM':
                eigenvals, eigenvecs = eigenvals[:k], eigenvecs[:, :k]
            else:
                eigenvals, eigenvecs = eigenvals[-k:], eigenvecs[:, -k:]
                
    except (ArpackError, linalg.LinAlgError) as e:
        warnings.warn(f"Eigendecomposition failed: {e}. Using fallback method.")
        # Fallback to regularized version
        regularized_matrix = matrix + 1e-6 * np.eye(n)
        eigenvals, eigenvecs = linalg.eigh(regularized_matrix)
        if which == 'SM':
            eigenvals, eigenvecs = eigenvals[:k], eigenvecs[:, :k]
        else:
            eigenvals, eigenvecs = eigenvals[-k:], eigenvecs[:, -k:]
    
    # Stability metrics
    condition_number = np.max(eigenvals) / (np.min(eigenvals) + 1e-12)
    
    # Spectral gap (difference between k-th and (k+1)-th eigenvalue)
    if len(eigenvals) > 1:
        spectral_gap = eigenvals[1] - eigenvals[0] if which == 'SM' else eigenvals[-1] - eigenvals[-2]
    else:
        spectral_gap = 0.0
    
    stability_metrics = {
        "condition_number": float(condition_number),
        "spectral_gap": float(abs(spectral_gap)),
        "eigenval_range": [float(np.min(eigenvals)), float(np.max(eigenvals))],
        "frobenius_norm": float(linalg.norm(matrix, 'fro')),
        "nuclear_norm": float(np.sum(np.abs(eigenvals))),
    }
    
    return eigenvals, eigenvecs, stability_metrics


def _spectral_clustering_rerank(candidates: List[Dict[str, Any]], 
                               embeddings: np.ndarray,
                               n_clusters: int = 5,
                               cluster_weight: float = 0.3) -> List[Dict[str, Any]]:
    """
    Rerank candidates using spectral clustering to promote semantic coherence.
    
    Args:
        candidates: List of candidate documents with scores
        embeddings: Document embeddings
        n_clusters: Number of spectral clusters
        cluster_weight: Weight for cluster-based reranking
        
    Returns:
        Reranked candidates with spectral scores
    """
    if len(candidates) == 0 or embeddings.size == 0:
        return candidates
    
    n_docs = len(candidates)
    n_clusters = min(n_clusters, n_docs)
    
    if n_clusters < 2:
        # No clustering needed
        for candidate in candidates:
            candidate["spectral_score"] = candidate.get("hybrid_score", 0.0)
        return candidates
    
    try:
        # Perform spectral clustering
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity='cosine',
            random_state=42,
            n_neighbors=min(10, n_docs - 1)
        )
        cluster_labels = spectral.fit_predict(embeddings)
        
        # Compute cluster quality scores
        cluster_scores = {}
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            if np.sum(cluster_mask) > 1:
                cluster_embeddings = embeddings[cluster_mask]
                # Intra-cluster similarity (higher is better)
                intra_sim = np.mean(cosine_similarity(cluster_embeddings))
                cluster_scores[cluster_id] = intra_sim
            else:
                cluster_scores[cluster_id] = 0.0
        
        # Rerank based on cluster membership and quality
        for i, candidate in enumerate(candidates):
            if i < len(cluster_labels):
                cluster_id = cluster_labels[i]
                cluster_quality = cluster_scores[cluster_id]
                
                # Combine original score with cluster quality
                original_score = candidate.get("hybrid_score", 0.0)
                spectral_bonus = cluster_weight * cluster_quality
                candidate["spectral_score"] = original_score + spectral_bonus
                candidate["cluster_id"] = int(cluster_id)
                candidate["cluster_quality"] = cluster_quality
            else:
                candidate["spectral_score"] = candidate.get("hybrid_score", 0.0)
                candidate["cluster_id"] = -1
                candidate["cluster_quality"] = 0.0
                
    except Exception as e:
        warnings.warn(f"Spectral clustering failed: {e}. Using original scores.")
        for candidate in candidates:
            candidate["spectral_score"] = candidate.get("hybrid_score", 0.0)
            candidate["cluster_id"] = -1
            candidate["cluster_quality"] = 0.0
    
    # Re-sort by spectral score
    candidates.sort(key=lambda x: -x["spectral_score"])
    
    return candidates


def _validate_eigenvalue_stability(eigenvals: np.ndarray, 
                                 matrix: np.ndarray,
                                 perturbation_scale: float = 1e-6) -> Dict[str, float]:
    """
    Validate eigenvalue stability using perturbation theory.
    
    Based on Weyl's inequalities: |λᵢ(A) - λᵢ(A + E)| ≤ ||E||₂
    
    Args:
        eigenvals: Computed eigenvalues
        matrix: Original matrix
        perturbation_scale: Scale of perturbation for testing
        
    Returns:
        Stability validation metrics
    """
    if eigenvals.size == 0 or matrix.size == 0:
        return {"weyl_bound": 0.0, "relative_error": 0.0, "stability_score": 1.0}
    
    n = matrix.shape[0]
    
    # Generate random perturbation
    np.random.seed(42)  # For reproducibility
    perturbation = perturbation_scale * np.random.randn(n, n)
    perturbation = (perturbation + perturbation.T) / 2  # Make symmetric
    
    # Compute perturbed eigenvalues
    try:
        perturbed_matrix = matrix + perturbation
        perturbed_eigenvals, _ = linalg.eigh(perturbed_matrix)
        
        # Match eigenvalues by sorting
        eigenvals_sorted = np.sort(eigenvals)
        perturbed_sorted = np.sort(perturbed_eigenvals)
        
        # Take same number of eigenvalues
        n_compare = min(len(eigenvals_sorted), len(perturbed_sorted))
        eigenvals_matched = eigenvals_sorted[:n_compare] 
        perturbed_matched = perturbed_sorted[:n_compare]
        
        # Compute stability metrics
        perturbation_norm = linalg.norm(perturbation, ord=2)
        eigenval_changes = np.abs(eigenvals_matched - perturbed_matched)
        
        weyl_bound = perturbation_norm  # Theoretical upper bound
        max_change = np.max(eigenval_changes) if len(eigenval_changes) > 0 else 0.0
        
        # Relative error
        eigenval_magnitudes = np.abs(eigenvals_matched) + 1e-12
        relative_errors = eigenval_changes / eigenval_magnitudes
        max_relative_error = np.max(relative_errors) if len(relative_errors) > 0 else 0.0
        
        # Stability score (1.0 = perfectly stable, 0.0 = unstable)
        stability_score = np.exp(-max_relative_error / 0.01)  # Exponential decay
        
        return {
            "weyl_bound": float(weyl_bound),
            "max_eigenval_change": float(max_change),
            "relative_error": float(max_relative_error),
            "stability_score": float(stability_score),
            "perturbation_norm": float(perturbation_norm)
        }
        
    except Exception as e:
        warnings.warn(f"Stability validation failed: {e}")
        return {
            "weyl_bound": float('inf'),
            "max_eigenval_change": float('inf'), 
            "relative_error": float('inf'),
            "stability_score": 0.0,
            "perturbation_norm": float(perturbation_scale)
        }


def _validate_spectral_properties(similarity_matrix: np.ndarray,
                                laplacian: np.ndarray,
                                eigenvals: np.ndarray) -> Dict[str, Any]:
    """
    Validate mathematical properties of spectral decomposition.
    
    Args:
        similarity_matrix: Original similarity matrix
        laplacian: Graph Laplacian matrix
        eigenvals: Computed eigenvalues
        
    Returns:
        Dictionary of validation results
    """
    validation_results = {}
    
    # 1. Symmetry check
    symmetry_error = linalg.norm(similarity_matrix - similarity_matrix.T, 'fro')
    validation_results["symmetry_error"] = float(symmetry_error)
    validation_results["is_symmetric"] = symmetry_error < 1e-10
    
    # 2. Positive semi-definiteness of similarity matrix
    try:
        sim_eigenvals = linalg.eigvals(similarity_matrix)
        min_sim_eigenval = np.min(np.real(sim_eigenvals))  # Take real part to avoid complex warning
        validation_results["similarity_psd"] = min_sim_eigenval >= -1e-10
        validation_results["similarity_min_eigenval"] = float(min_sim_eigenval)
    except:
        validation_results["similarity_psd"] = False
        validation_results["similarity_min_eigenval"] = float('nan')
    
    # 3. Laplacian properties
    # - Should have at least one zero eigenvalue
    # - All eigenvalues should be non-negative
    min_laplacian_eigenval = np.min(eigenvals) if len(eigenvals) > 0 else 0.0
    has_zero_eigenval = np.abs(min_laplacian_eigenval) < 1e-8
    
    validation_results["laplacian_min_eigenval"] = float(min_laplacian_eigenval)  
    validation_results["has_zero_eigenval"] = has_zero_eigenval
    validation_results["laplacian_psd"] = min_laplacian_eigenval >= -1e-10
    
    # 4. Number of connected components (number of zero eigenvalues)
    zero_eigenvals = np.sum(np.abs(eigenvals) < 1e-8)
    validation_results["n_zero_eigenvals"] = int(zero_eigenvals)
    validation_results["n_connected_components"] = int(zero_eigenvals)
    
    # 5. Spectral gap analysis
    if len(eigenvals) > 1:
        eigenvals_sorted = np.sort(eigenvals)
        gaps = np.diff(eigenvals_sorted)
        max_gap = np.max(gaps) if len(gaps) > 0 else 0.0
        validation_results["max_spectral_gap"] = float(max_gap)
        validation_results["gap_ratio"] = float(max_gap / (eigenvals_sorted[-1] + 1e-12))
    else:
        validation_results["max_spectral_gap"] = 0.0
        validation_results["gap_ratio"] = 0.0
    
    # 6. Matrix conditioning
    condition_number = np.max(eigenvals) / (np.min(eigenvals[eigenvals > 1e-10]) + 1e-12)
    validation_results["condition_number"] = float(condition_number)
    validation_results["is_well_conditioned"] = condition_number < 1e12
    
    # 7. Numerical precision check
    reconstruction_error = 0.0
    try:
        # Basic reconstruction check for small matrices
        if laplacian.shape[0] < 500:
            full_eigenvals, full_eigenvecs = linalg.eigh(laplacian)
            reconstructed = full_eigenvecs @ np.diag(full_eigenvals) @ full_eigenvecs.T
            reconstruction_error = linalg.norm(laplacian - reconstructed, 'fro')
    except:
        reconstruction_error = float('inf')
    
    validation_results["reconstruction_error"] = float(reconstruction_error)
    validation_results["numerical_stability"] = reconstruction_error < 1e-10
    
    return validation_results


def process(data: Any, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Main processing function for spectral graph theory retrieval enhancement.
    
    Args:
        data: Input data containing candidates and embeddings
        context: Processing context with parameters
        
    Returns:
        Enhanced data with spectral reranking and mathematical validation
    """
    ctx = context or {}
    debug = bool(ctx.get("debug", False))
    trace_log: List[Dict[str, Any]] = []
    
# # #     # Extract candidates and embeddings from input  # Module not found  # Module not found  # Module not found
    candidates = []
    embeddings = np.zeros((0, 0))
    
    if isinstance(data, dict):
        candidates = data.get("candidates", [])
        
        # Try multiple sources for embeddings
        if "vector_index" in data and data["vector_index"]:
            embeddings = np.array(data["vector_index"])
        elif "projections" in data and "svd" in data["projections"]:
            embeddings = np.array(data["projections"]["svd"])
        elif "embeddings" in data:
            embeddings = np.array(data["embeddings"])
        
        if debug:
            trace_log.append({
                "step": "input_extraction",
                "n_candidates": len(candidates),
                "embedding_shape": embeddings.shape if embeddings.size > 0 else (0, 0)
            })
    
    # Skip processing if insufficient data
    if len(candidates) < 2 or embeddings.size == 0:
        if debug:
            trace_log.append({"step": "skip", "reason": "insufficient_data"})
        
        return {
            **(data if isinstance(data, dict) else {}),
            "spectral_metrics": {"status": "skipped", "reason": "insufficient_data"},
            "spectral_explain_plan": {"method": "none", "status": "skipped"},
            "trace": trace_log if debug else []
        }
    
    # Ensure embeddings match candidates
    n_candidates = len(candidates)
    if embeddings.shape[0] != n_candidates:
        # Pad or truncate embeddings to match candidates
        if embeddings.shape[0] > n_candidates:
            embeddings = embeddings[:n_candidates]
        else:
            # Pad with mean embedding
            mean_embedding = np.mean(embeddings, axis=0) if embeddings.size > 0 else np.zeros(64)
            padding_needed = n_candidates - embeddings.shape[0]
            padding = np.tile(mean_embedding, (padding_needed, 1))
            embeddings = np.vstack([embeddings, padding])
    
    # Initialize Hyperbolic Tensor Networks
    hyperbolic_tensor_net = HyperbolicTensorNetworks(
        embedding_dim=embeddings.shape[1],
        tensor_rank=ctx.get("tensor_rank", 8),
        regularization=regularization
    )
    
    # 1. Construct document similarity graph
    similarity_method = ctx.get("similarity_method", "cosine")
    k_nearest = ctx.get("k_nearest_neighbors", min(10, n_candidates - 1))
    
    similarity_matrix = _compute_document_similarity_graph(
        embeddings, 
        method=similarity_method,
        k_nearest=k_nearest
    )
    
    if debug:
        trace_log.append({
            "step": "similarity_graph",
            "method": similarity_method,
            "matrix_shape": similarity_matrix.shape,
            "sparsity": float(np.sum(similarity_matrix == 0)) / similarity_matrix.size
        })
    
    # 2. Compute graph Laplacian
    use_normalized = ctx.get("normalized_laplacian", True)
    regularization = ctx.get("regularization", 1e-8)
    
    laplacian, degrees = _compute_graph_laplacian(
        similarity_matrix,
        normalized=use_normalized,
        regularization=regularization
    )
    
    # 3. Eigenvalue decomposition  
    n_eigenvals = ctx.get("n_eigenvalues", min(10, n_candidates - 1))
    eigenvals, eigenvecs, stability_metrics = _compute_eigendecomposition(
        laplacian, k=n_eigenvals, which='SM'
    )
    
    # 4. HYPERBOLIC TENSOR NETWORK ENHANCEMENT
    hyperbolic_enhancement = hyperbolic_tensor_net.enhance_retrieval_with_hyperbolic_tensors(
        embeddings=embeddings,
        similarity_matrix=similarity_matrix,
        eigenvalues=eigenvals,
        eigenvectors=eigenvecs
    )
    
    # Use enhanced similarity matrix if available
    if "hybrid_similarity" in hyperbolic_enhancement and hyperbolic_enhancement["hybrid_similarity"].size > 0:
        enhanced_similarity = hyperbolic_enhancement["hybrid_similarity"]
        # Recompute Laplacian with hybrid similarity
        enhanced_laplacian, _ = _compute_graph_laplacian(
            enhanced_similarity,
            normalized=use_normalized,
            regularization=regularization
        )
        # Use enhanced eigenvalues if available
        if "enhanced_eigenvalues" in hyperbolic_enhancement:
            eigenvals = hyperbolic_enhancement["enhanced_eigenvalues"]
            eigenvecs = hyperbolic_enhancement["enhanced_eigenvectors"]
    else:
        enhanced_similarity = similarity_matrix
    
    if debug:
        trace_log.append({
            "step": "eigendecomposition",
            "n_eigenvals": len(eigenvals),
            "eigenval_range": [float(np.min(eigenvals)), float(np.max(eigenvals))],
            "stability": stability_metrics
        })
        
        trace_log.append({
            "step": "hyperbolic_tensor_enhancement",
            "hyperbolic_metrics": hyperbolic_enhancement.get("hyperbolic_metrics", {}),
            "tensor_compression": hyperbolic_enhancement.get("hyperbolic_metrics", {}).get("tensor_compression", {}),
            "enhanced_similarity_shape": hyperbolic_enhancement.get("hybrid_similarity", np.array([])).shape
        })
    
    # 5. Spectral clustering reranking (using enhanced similarity if available)
    n_clusters = ctx.get("n_spectral_clusters", min(5, n_candidates))
    cluster_weight = ctx.get("cluster_rerank_weight", 0.3)
    
    # Use hyperbolic embeddings for clustering if available
    clustering_embeddings = hyperbolic_enhancement.get("hyperbolic_embeddings", embeddings)
    
    reranked_candidates = _spectral_clustering_rerank(
        candidates.copy(),  # Don't modify original
        clustering_embeddings,
        n_clusters=n_clusters,
        cluster_weight=cluster_weight
    )
    
    # 6. Mathematical validation
    stability_validation = _validate_eigenvalue_stability(eigenvals, laplacian)
    spectral_validation = _validate_spectral_properties(enhanced_similarity, laplacian, eigenvals)
    
    # Combine all validation results
    all_validations = {**stability_validation, **spectral_validation}
    
    # 7. Compute enhancement metrics
    original_scores = [c.get("hybrid_score", 0.0) for c in candidates]
    spectral_scores = [c.get("spectral_score", 0.0) for c in reranked_candidates]
    
    # Rank correlation between original and spectral rankings
    original_ranks = np.argsort([-score for score in original_scores])
    spectral_ranks = np.argsort([-score for score in spectral_scores])
    
    rank_changes = np.sum(original_ranks != spectral_ranks)
    
    enhancement_metrics = {
        "similarity_method": similarity_method,
        "laplacian_type": "normalized" if use_normalized else "unnormalized",
        "n_eigenvalues": len(eigenvals),
        "n_clusters": n_clusters,
        "rank_changes": int(rank_changes),
        "rank_change_ratio": float(rank_changes) / n_candidates,
        "spectral_gap": stability_metrics.get("spectral_gap", 0.0),
        "condition_number": stability_metrics.get("condition_number", 1.0),
        "hyperbolic_tensor_enhancement": hyperbolic_enhancement.get("hyperbolic_metrics", {}),
        "tensor_compression_ratio": hyperbolic_enhancement.get("hyperbolic_metrics", {}).get("tensor_compression", {}).get("compression_ratio", 1.0),
        "hyperbolic_features": hyperbolic_enhancement.get("hyperbolic_features", {}),
        "validation_passed": all([
            all_validations.get("is_symmetric", False),
            all_validations.get("laplacian_psd", False),
            all_validations.get("is_well_conditioned", False),
            stability_validation.get("stability_score", 0.0) > 0.5,
            hyperbolic_enhancement.get("hyperbolic_metrics", {}).get("status") == "success"
        ])
    }
    
    # Publish metrics
    try:
        publish_metric("retrieval.spectral", {
            "metrics": enhancement_metrics,
            "validation": all_validations,
            "stage": ctx.get("stage", "spectral_enhancement")
        })
    except Exception:
        pass
    
    # Build explain plan
    explain_plan = {
        "method": "Spectral Graph Theory with Hyperbolic Tensor Networks Enhancement",
        "similarity_construction": f"{similarity_method} similarity with k={k_nearest} neighbors",
        "laplacian_type": "normalized" if use_normalized else "unnormalized",  
        "eigendecomposition": f"Computed {len(eigenvals)} smallest eigenvalues",
        "spectral_clustering": f"Used {n_clusters} clusters for reranking",
        "hyperbolic_enhancement": {
            "poincare_embeddings": hyperbolic_enhancement.get("hyperbolic_metrics", {}).get("status", "unknown"),
            "tensor_decomposition": f"Rank-{hyperbolic_tensor_net.tensor_rank} tensor networks",
            "compression_ratio": hyperbolic_enhancement.get("hyperbolic_metrics", {}).get("tensor_compression", {}).get("compression_ratio", 1.0),
            "hyperbolic_distance_metric": "Poincaré disk model"
        },
        "validation_status": "passed" if enhancement_metrics["validation_passed"] else "failed",
        "numerical_stability": "stable" if stability_validation.get("stability_score", 0) > 0.5 else "unstable"
    }
    
    # Build output
    out: Dict[str, Any] = {}
    if isinstance(data, dict):
        out.update(data)
    
    out.update({
        "candidates": reranked_candidates,  # Enhanced candidates with spectral scores
        "spectral_similarity_matrix": similarity_matrix.tolist(),
        "hyperbolic_similarity_matrix": hyperbolic_enhancement.get("hyperbolic_similarity", np.array([])).tolist(),
        "hybrid_similarity_matrix": hyperbolic_enhancement.get("hybrid_similarity", similarity_matrix).tolist(),
        "spectral_eigenvalues": eigenvals.tolist(),
        "spectral_eigenvectors": eigenvecs.tolist() if eigenvecs.size > 0 else [],
        "hyperbolic_embeddings": hyperbolic_enhancement.get("hyperbolic_embeddings", np.array([])).tolist(),
        "tensor_decomposition": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in hyperbolic_enhancement.get("tensor_decomposition", {}).items()},
        "hyperbolic_features": hyperbolic_enhancement.get("hyperbolic_features", {}),
        "spectral_metrics": enhancement_metrics,
        "spectral_validation": all_validations,
        "spectral_explain_plan": explain_plan
    })
    
    if debug:
        out["trace"] = trace_log
    
    # Optional persistence
    try:
        data_dir = os.path.join(os.getcwd(), "data")
        if os.path.isdir(data_dir):
            import json
            
            # Save spectral analysis results with hyperbolic enhancement
            spectral_results = {
                "eigenvalues": eigenvals.tolist(),
                "metrics": enhancement_metrics,
                "validation": all_validations,
                "explain_plan": explain_plan,
                "hyperbolic_enhancement": {
                    "metrics": hyperbolic_enhancement.get("hyperbolic_metrics", {}),
                    "features": hyperbolic_enhancement.get("hyperbolic_features", {}),
                    "tensor_compression": hyperbolic_enhancement.get("hyperbolic_metrics", {}).get("tensor_compression", {})
                }
            }
            
            with open(os.path.join(data_dir, "spectral_enhancement.json"), "w", encoding="utf-8") as f:
                json.dump(spectral_results, f, indent=2)
                
    except Exception:
        pass
    
    return out


if __name__ == "__main__":
    # Demonstration of spectral enhancement
    import numpy as np
    
    # Mock data for testing
    test_embeddings = np.random.randn(8, 32)  # 8 documents, 32-dim embeddings
    test_candidates = [
        {"doc_id": i, "hybrid_score": np.random.random()}
        for i in range(8)
    ]
    
    test_data = {
        "candidates": test_candidates,
        "vector_index": test_embeddings.tolist()
    }
    
    # Process with spectral enhancement
    result = process(test_data, {"debug": True, "n_spectral_clusters": 3})
    
    print("Spectral Enhancement Results:")
    print(f"- Processed {len(result['candidates'])} candidates")
    print(f"- Computed {len(result['spectral_eigenvalues'])} eigenvalues")
    print(f"- Validation passed: {result['spectral_metrics']['validation_passed']}")
    print(f"- Rank changes: {result['spectral_metrics']['rank_changes']}")
    
    # Show top candidate changes
    original_top = max(test_candidates, key=lambda x: x['hybrid_score'])
    spectral_top = result['candidates'][0]
    
    print(f"- Original top candidate: {original_top['doc_id']} (score: {original_top['hybrid_score']:.3f})")
    print(f"- Spectral top candidate: {spectral_top['doc_id']} (score: {spectral_top['spectral_score']:.3f})")
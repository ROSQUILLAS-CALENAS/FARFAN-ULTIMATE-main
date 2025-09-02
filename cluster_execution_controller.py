"""
Cluster Execution Controller
- Enforces/validates that the questionnaire is applied four times (clusters C1..C4).
- Builds a deterministic micro-level container per cluster when inputs are present.
- Degrades gracefully to a structured pass-through with explicit gap flags.

Entry: process(data, context) -> merged dict with 'cluster_audit'.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

# Import evidence processor for integration
try:
    from evidence_processor import (
        EvidenceProcessor,
        StructuredEvidence, 
        EvidenceChunk,
        SourceMetadata,
        EvidenceType,
        ScoringMetrics
    )
    EVIDENCE_PROCESSOR_AVAILABLE = True
except ImportError:
    EVIDENCE_PROCESSOR_AVAILABLE = False
    EvidenceProcessor = None

REQUIRED_CLUSTERS = ("C1", "C2", "C3", "C4")


class ClusterCountError(Exception):
    """Exception raised when cluster count validation fails."""
    pass


def _stable_hash(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def validate_cluster_count(clusters: List[str]) -> None:
    """
    Validates that exactly four clusters (C1-C4) are present.
    
    Args:
        clusters: List of cluster identifiers
        
    Raises:
        ClusterCountError: If the count differs from exactly 4 clusters
    """
    required_set = set(REQUIRED_CLUSTERS)
    clusters_set = set(clusters)
    
    if len(clusters) != 4:
        raise ClusterCountError(
            f"Expected exactly 4 clusters, got {len(clusters)}: {clusters}"
        )
    
    if clusters_set != required_set:
        missing = required_set - clusters_set
        extra = clusters_set - required_set
        error_msg = f"Cluster validation failed. Required: {list(required_set)}, Got: {clusters}"
        if missing:
            error_msg += f". Missing: {list(missing)}"
        if extra:
            error_msg += f". Extra: {list(extra)}"
        raise ClusterCountError(error_msg)


def _apply_questionnaire_to_cluster(cluster_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the human rights questionnaire to a specific cluster for DNP alignment."""
    
    # Decálogo de Derechos Humanos questionnaire structure
    decalogo_questions = {
        "P1": "¿Se garantiza el derecho a la vida y seguridad de la población?",
        "P2": "¿Se respeta la dignidad humana en todos los procesos?", 
        "P3": "¿Se asegura la igualdad y no discriminación?",
        "P4": "¿Se promueve la participación ciudadana efectiva?",
        "P5": "¿Se garantiza el acceso a servicios básicos?",
        "P6": "¿Se protege el medio ambiente y recursos naturales?",
        "P7": "¿Se fomenta el desarrollo económico inclusivo?",
        "P8": "¿Se respetan los derechos culturales y territoriales?",
        "P9": "¿Se garantiza el acceso a la justicia?",
        "P10": "¿Se promueve la transparencia y rendición de cuentas?"
    }
    
    # Apply questionnaire to cluster data
    cluster_evaluation = {
        "cluster_id": cluster_id,
        "questions_applied": len(decalogo_questions),
        "evaluation_results": {}
    }
    
    # Extract evidence from cluster data
    evidence_data = data.get("evidence", {})
    
    for question_id, question_text in decalogo_questions.items():
        # Simulate evidence linking based on cluster data
        relevant_evidence = []
        
        # Look for evidence related to this question in the cluster
        if isinstance(evidence_data, dict):
            for evidence_id, evidence_content in evidence_data.items():
                if isinstance(evidence_content, dict):
                    evidence_score = evidence_content.get("relevance_score", 0.0)
                    if evidence_score > 0.5:  # Threshold for relevance
                        relevant_evidence.append({
                            "evidence_id": evidence_id,
                            "relevance_score": evidence_score,
                            "content_summary": evidence_content.get("summary", "")
                        })
        
        cluster_evaluation["evaluation_results"][question_id] = {
            "question": question_text,
            "evidence_count": len(relevant_evidence),
            "evidence_ids": [e["evidence_id"] for e in relevant_evidence],
            "average_relevance": sum(e["relevance_score"] for e in relevant_evidence) / len(relevant_evidence) if relevant_evidence else 0.0,
            "compliance_indicator": "CUMPLE" if len(relevant_evidence) >= 2 else "CUMPLE_PARCIAL" if len(relevant_evidence) == 1 else "NO_CUMPLE"
        }
    
    return cluster_evaluation


def _link_evidence_to_questions(cluster_results: Dict[str, Any]) -> Dict[str, Any]:
    """Link evidence to specific questions across all clusters for traceability."""
    
    evidence_links = {
        "total_evidence_items": 0,
        "linked_evidence": {},
        "cross_cluster_references": {},
        "traceability_matrix": {}
    }
    
    for cluster_id, cluster_data in cluster_results.items():
        if "evaluation_results" in cluster_data:
            for question_id, question_data in cluster_data["evaluation_results"].items():
                evidence_ids = question_data.get("evidence_ids", [])
                
                for evidence_id in evidence_ids:
                    if evidence_id not in evidence_links["linked_evidence"]:
                        evidence_links["linked_evidence"][evidence_id] = []
                    
                    evidence_links["linked_evidence"][evidence_id].append({
                        "cluster_id": cluster_id,
                        "question_id": question_id,
                        "relevance_score": question_data.get("average_relevance", 0.0)
                    })
                    
                    # Update traceability matrix
                    matrix_key = f"{cluster_id}_{question_id}"
                    if matrix_key not in evidence_links["traceability_matrix"]:
                        evidence_links["traceability_matrix"][matrix_key] = []
                    evidence_links["traceability_matrix"][matrix_key].append(evidence_id)
    
    evidence_links["total_evidence_items"] = len(evidence_links["linked_evidence"])
    
    # Identify cross-cluster evidence references
    for evidence_id, references in evidence_links["linked_evidence"].items():
        clusters_referenced = set(ref["cluster_id"] for ref in references)
        if len(clusters_referenced) > 1:
            evidence_links["cross_cluster_references"][evidence_id] = list(clusters_referenced)
    
    return evidence_links


def process_clusters_sequentially(data: Dict[str, Any], context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Process clusters C1-C4 in fixed sequential order, applying questionnaires deterministically
    and capturing evidence_ids for each micro-level response.
    
    Args:
        data: Input data containing cluster information and evidence
        context: Optional context for processing
        
    Returns:
        Dict containing cluster processing results with evidence linking
    """
    ctx = context or {}
    evidence_processor = None
    
    # Initialize evidence processor if available
    if EVIDENCE_PROCESSOR_AVAILABLE:
        evidence_processor = EvidenceProcessor()
    
    # Determine clusters present
    clusters_input = data.get("clusters") or ctx.get("clusters", [])
    clusters = [c for c in clusters_input if isinstance(c, str) and c.strip()]
    
    # Validate cluster count
    try:
        validate_cluster_count(clusters)
        cluster_validation_status = "valid"
    except ClusterCountError as e:
        cluster_validation_status = f"invalid: {str(e)}"
    
    # Sequential processing results
    sequential_results = {
        "cluster_validation": cluster_validation_status,
        "processing_order": list(REQUIRED_CLUSTERS),
        "cluster_results": {},
        "evidence_tracking": {},
        "micro_level_responses": {},
        "processing_metadata": {
            "total_clusters": len(REQUIRED_CLUSTERS),
            "processed_clusters": 0,
            "evidence_ids_generated": []
        }
    }
    
    # Process clusters in fixed order C1 -> C2 -> C3 -> C4
    for cluster_id in REQUIRED_CLUSTERS:
        cluster_data = {
            "cluster_id": cluster_id,
            "processing_order": list(REQUIRED_CLUSTERS).index(cluster_id) + 1,
            "questionnaire_applied": False,
            "evidence_ids": [],
            "micro_responses": []
        }
        
        if cluster_id in clusters:
            # Apply questionnaire to cluster
            questionnaire_result = _apply_questionnaire_to_cluster(cluster_id, data)
            cluster_data.update(questionnaire_result)
            cluster_data["questionnaire_applied"] = True
            
            # Process micro-level responses with evidence linking
            cluster_answers = data.get("cluster_answers", {}).get(cluster_id, [])
            
            for answer_idx, answer in enumerate(cluster_answers):
                if not isinstance(answer, dict):
                    continue
                
                question_id = answer.get("question_id") or f"Q{answer_idx + 1}"
                response_text = answer.get("response") or answer.get("synthesized_answers", "")
                
                # Generate evidence ID for this micro-level response
                evidence_id = f"ev_{cluster_id}_{question_id}_{_stable_hash(response_text)[:8]}"
                
                micro_response = {
                    "question_id": question_id,
                    "evidence_id": evidence_id,
                    "cluster_id": cluster_id,
                    "response_text": response_text,
                    "answer_index": answer_idx,
                    "linked_evidence": answer.get("evidence_ids", []),
                    "citations": answer.get("citations", [])
                }
                
                cluster_data["evidence_ids"].append(evidence_id)
                cluster_data["micro_responses"].append(micro_response)
                sequential_results["processing_metadata"]["evidence_ids_generated"].append(evidence_id)
                
                # Track evidence with cluster and question linkage
                if evidence_id not in sequential_results["evidence_tracking"]:
                    sequential_results["evidence_tracking"][evidence_id] = []
                
                sequential_results["evidence_tracking"][evidence_id].append({
                    "cluster_id": cluster_id,
                    "question_id": question_id,
                    "answer_index": answer_idx,
                    "response_length": len(response_text) if response_text else 0,
                    "has_citations": len(answer.get("citations", [])) > 0
                })
                
                # Process with evidence_processor if available
                if evidence_processor and response_text:
                    try:
                        # Create evidence chunk from response
                        evidence_chunk = EvidenceChunk(
                            chunk_id=f"chunk_{cluster_id}_{question_id}",
                            text=response_text[:500],  # Truncate for processing
                            processing_timestamp=None  # Will use default
                        )
                        
                        # Create basic source metadata
                        source_metadata = SourceMetadata(
                            document_id=f"cluster_{cluster_id}",
                            title=f"Cluster {cluster_id} Response",
                            author="DNP System",
                            document_type="cluster_response"
                        )
                        
                        # Process evidence
                        structured_evidence = evidence_processor.process_evidence_chunks(
                            chunks=[evidence_chunk],
                            metadata_list=[source_metadata],
                            question_id=question_id,
                            dimension=cluster_id,
                            evidence_type=EvidenceType.DIRECT_QUOTE
                        )
                        
                        if structured_evidence:
                            micro_response["structured_evidence"] = {
                                "evidence_id": structured_evidence[0].evidence_id,
                                "validation_result": structured_evidence[0].validation_result,
                                "scoring": structured_evidence[0].scoring,
                                "traceability_path": structured_evidence[0].get_traceability_path()
                            }
                            
                    except Exception as e:
                        micro_response["evidence_processing_error"] = str(e)
                
                # Store micro-level response
                if cluster_id not in sequential_results["micro_level_responses"]:
                    sequential_results["micro_level_responses"][cluster_id] = []
                sequential_results["micro_level_responses"][cluster_id].append(micro_response)
            
            sequential_results["processing_metadata"]["processed_clusters"] += 1
        else:
            # Mark cluster as missing
            cluster_data["status"] = "missing"
            cluster_data["reason"] = f"Cluster {cluster_id} not found in input data"
        
        sequential_results["cluster_results"][cluster_id] = cluster_data
    
    # Generate cross-cluster evidence links
    cross_cluster_links = {}
    for evidence_id, tracking_info in sequential_results["evidence_tracking"].items():
        clusters_referenced = set(info["cluster_id"] for info in tracking_info)
        if len(clusters_referenced) > 1:
            cross_cluster_links[evidence_id] = {
                "clusters": list(clusters_referenced),
                "total_references": len(tracking_info),
                "questions_involved": list(set(info["question_id"] for info in tracking_info))
            }
    
    sequential_results["cross_cluster_links"] = cross_cluster_links
    sequential_results["processing_metadata"]["cross_cluster_evidence_count"] = len(cross_cluster_links)
    
    return sequential_results


def process(data: Any, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    ctx = context or {}
    out: Dict[str, Any] = {}
    if isinstance(data, dict):
        out.update(data)

    # Determine clusters present from input or context
    clusters_input = []
    if isinstance(data, dict):
        clusters_input = data.get("clusters") or []
    if not clusters_input and isinstance(ctx.get("clusters"), list):
        clusters_input = ctx.get("clusters")
    # Normalize
    clusters = [c for c in clusters_input if isinstance(c, str) and c.strip()]
    required = list(REQUIRED_CLUSTERS)
    missing = [c for c in required if c not in clusters]
    
    # Validate cluster count if all clusters are present
    cluster_count_valid = True
    cluster_validation_error = None
    if len(clusters) > 0:
        try:
            validate_cluster_count(clusters)
        except ClusterCountError as e:
            cluster_count_valid = False
            cluster_validation_error = str(e)

    # Build micro structure if we have per-cluster answers
    micro: Dict[str, Dict[str, Any]] = {}
    if isinstance(data, dict):
        cluster_answers = data.get("cluster_answers")  # expected: {C#: [{...}, ...]}
        if isinstance(cluster_answers, dict):
            for c in required:
                answers = cluster_answers.get(c) or []
                if isinstance(answers, list):
                    # Each answer item may contain question_id and synthesized_answers entry
                    micro[c] = {
                        "answers": answers,
                        "count": len(answers),
                        "evidence_linked": all(
                            isinstance(a, dict)
                            and any(a.get("evidence_ids") or a.get("citations") or [])
                            for a in answers
                        ),
                    }

    # Build audit summary
    cluster_audit = {
        "required": required,
        "present": clusters,
        "missing": missing,
        "complete": len(missing) == 0 and all(
            (micro.get(c, {}).get("count", 0) > 0) for c in required
        ),
        "non_redundant": True,  # deterministic assumption unless duplicates detected
        "hash": _stable_hash("|".join(sorted(clusters)) or "none"),
        "gaps": [],
        "micro": micro,
    }

    # Detect duplicates if cluster_answers contains repeated question_ids in same cluster
    dups = []
    for c, payload in micro.items():
        seen = set()
        for a in payload.get("answers", []):
            qid = a.get("question_id") or a.get("question")
            if not qid:
                continue
            if qid in seen:
                dups.append((c, qid))
            else:
                seen.add(qid)
    if dups:
        cluster_audit["non_redundant"] = False
        cluster_audit["gaps"].append("duplicate_items")

    if missing:
        cluster_audit["gaps"].append("missing_clusters")
    if not micro:
        cluster_audit["gaps"].append("no_micro_answers")

    # Execute four-cluster processing workflow (C1-C4) with deterministic sequential processing
    sequential_processing_results = process_clusters_sequentially(data, context)
    
    # Execute legacy cluster processing for compatibility
    cluster_processing_results = {}
    evidence_tracking = {}
    
    for cluster_id in REQUIRED_CLUSTERS:
        if cluster_id in clusters:
            # Apply questionnaire to this cluster
            cluster_evaluation = _apply_questionnaire_to_cluster(cluster_id, data)
            cluster_processing_results[cluster_id] = cluster_evaluation
            
            # Track evidence for this cluster
            if "evaluation_results" in cluster_evaluation:
                for question_id, question_data in cluster_evaluation["evaluation_results"].items():
                    evidence_ids = question_data.get("evidence_ids", [])
                    for evidence_id in evidence_ids:
                        if evidence_id not in evidence_tracking:
                            evidence_tracking[evidence_id] = []
                        evidence_tracking[evidence_id].append({
                            "cluster_id": cluster_id,
                            "question_id": question_id
                        })
        else:
            # Create placeholder for missing cluster
            cluster_processing_results[cluster_id] = {
                "cluster_id": cluster_id,
                "status": "missing",
                "questions_applied": 0,
                "evaluation_results": {}
            }
    
    # Link evidence across clusters
    evidence_links = _link_evidence_to_questions(cluster_processing_results)
    
    # Update cluster audit with processing results
    cluster_audit.update({
        "cluster_count_validation": {
            "valid": cluster_count_valid,
            "error": cluster_validation_error,
            "required_clusters": list(REQUIRED_CLUSTERS),
            "actual_clusters": clusters
        },
        "sequential_processing": sequential_processing_results,
        "cluster_processing": cluster_processing_results,
        "evidence_tracking": evidence_tracking,
        "evidence_links": evidence_links,
        "questionnaire_coverage": {
            cluster_id: len(result.get("evaluation_results", {})) 
            for cluster_id, result in cluster_processing_results.items()
        },
        "total_questions_applied": sum(
            len(result.get("evaluation_results", {})) 
            for result in cluster_processing_results.values()
        ),
        "evidence_processor_integration": EVIDENCE_PROCESSOR_AVAILABLE,
        "deterministic_execution": {
            "clusters_processed_in_order": list(REQUIRED_CLUSTERS),
            "evidence_ids_generated": sequential_processing_results.get("processing_metadata", {}).get("evidence_ids_generated", []),
            "cross_cluster_evidence_count": sequential_processing_results.get("processing_metadata", {}).get("cross_cluster_evidence_count", 0)
        }
    })

    out["cluster_audit"] = cluster_audit
    return out

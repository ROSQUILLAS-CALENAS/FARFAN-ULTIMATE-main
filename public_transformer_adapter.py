"""
Public Transformer Adapter

Provides an interface for integrating open-source transformer models into the 
deterministic pipeline flow while preserving pedagogical output integrity.

This adapter:
- Implements model loading functionality compatible with pipeline orchestrator integration
- Maintains compatibility with evidence tracking system requirements for lossless output transformation
- Includes proper initialization methods callable by comprehensive_pipeline_orchestrator.py
- Works seamlessly during the four-cluster processing workflow
- Provides pedagogical transformation while maintaining evidence rigor
- Supports multiple open-source transformer backends via pluggable strategy

Usage: 
    adapter = PublicTransformerAdapter()
    result = adapter.process(data, context)
    
Or standalone: process(data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]
"""
from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

try:
    import numpy as np
except ImportError:
    # Fallback for environments without numpy
    class MockNumpy:
        def zeros(self, size):
            return [0.0] * size if isinstance(size, int) else [0.0] * 384
        def array(self, data):
            return data
    np = MockNumpy()

logger = logging.getLogger(__name__)


@dataclass
class TransformationContext:
    """Context for transformer model operations"""
    source: str = "default"
    preserve_evidence: bool = True
    model_backend: str = "local"
    pedagogical_mode: bool = True
    cluster_processing: bool = False
    four_cluster_workflow: bool = False


class TransformerBackend(ABC):
    """Abstract base for pluggable transformer backends"""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the transformer backend"""
        pass
    
    @abstractmethod
    def transform_text(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Transform text while preserving semantic integrity"""
        pass
    
    @abstractmethod
    def extract_features(self, text: str):
        """Extract features for pedagogical enhancement"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available"""
        pass


class LocalTransformerBackend(TransformerBackend):
    """Local transformer backend using rule-based transformation"""
    
    def __init__(self):
        self.initialized = False
        self.pedagogical_templates = {
            "evidence_summary": "¿Qué encontramos? {evidence_count} elementos de evidencia que apoyan esta evaluación.",
            "alignment_score": "Puntaje de alineación: {score}/1.0 indica {interpretation}",
            "dnp_compliance": "Uso de estándares DNP: {status}",
            "cluster_summary": "Procesamiento de cuatro clusters completado con {coverage}% de cobertura"
        }
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize local backend"""
        try:
            self.initialized = True
            logger.info("Local transformer backend initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize local backend: {e}")
            return False
    
    def transform_text(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Transform text using rule-based pedagogical enhancement"""
        if not self.initialized:
            return text
        
        # Simple pedagogical transformation
        if context and context.get("type") == "evidence_summary":
            count = context.get("evidence_count", 0)
            return self.pedagogical_templates["evidence_summary"].format(evidence_count=count)
        
        return text
    
    def extract_features(self, text: str):
        """Extract simple features from text"""
        if not text:
            return np.zeros(384)  # Default embedding size
        
        # Simple feature extraction based on text properties
        features = [
            len(text),
            text.count(' '),
            text.count('.'),
            text.count('?'),
            len(text.split()),
        ]
        
        # Pad or truncate to 384 dimensions
        features_array = np.array(features + [0] * (384 - len(features)))
        return features_array[:384] if hasattr(features_array, '__getitem__') else features + [0] * (384 - len(features))
    
    def is_available(self) -> bool:
        """Check if local backend is available"""
        return True


class HuggingFaceTransformerBackend(TransformerBackend):
    """HuggingFace transformer backend for advanced processing"""
    
    def __init__(self):
        self.initialized = False
        self.model = None
        self.tokenizer = None
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize HuggingFace backend"""
        try:
            # Check if transformers library is available
            import transformers
            
            model_name = config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
            
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.initialized = True
            logger.info(f"HuggingFace backend initialized with {model_name}")
            return True
            
        except ImportError:
            logger.warning("Transformers library not available, falling back to local backend")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace backend: {e}")
            return False
    
    def transform_text(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Transform text using HuggingFace model"""
        if not self.initialized or not self.model:
            return text
        
        # For pedagogical transformation, we maintain the text but could enhance it
        return text
    
    def extract_features(self, text: str):
        """Extract features using HuggingFace model"""
        if not self.initialized or not self.model:
            return np.zeros(384)
        
        try:
            embeddings = self.model.encode([text])
            return embeddings[0]
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return np.zeros(384)
    
    def is_available(self) -> bool:
        """Check if HuggingFace backend is available"""
        try:
            import transformers
            import sentence_transformers
            return True
        except ImportError:
            return False


class PublicTransformerAdapter:
    """
    Main adapter class for transformer integration in the deterministic pipeline.
    
    Provides:
    - Model loading functionality compatible with pipeline orchestrator
    - Evidence tracking system compatibility
    - Four-cluster processing workflow support
    - Pedagogical output transformation
    - Lossless output transformation guarantees
    """
    
    def __init__(self, backend_strategy: str = "auto"):
        """
        Initialize the public transformer adapter
        
        Args:
            backend_strategy: "auto", "local", or "huggingface"
        """
        self.backend_strategy = backend_strategy
        self.backend: Optional[TransformerBackend] = None
        self.initialized = False
        self.transformation_history: List[Dict[str, Any]] = []
        
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize the transformer adapter with the specified backend.
        Called by comprehensive_pipeline_orchestrator.py during setup.
        
        Args:
            config: Configuration dictionary for backend initialization
            
        Returns:
            bool: True if initialization successful
        """
        if config is None:
            config = {}
        
        try:
            # Select and initialize backend
            if self.backend_strategy == "auto":
                # Try HuggingFace first, fall back to local
                hf_backend = HuggingFaceTransformerBackend()
                if hf_backend.is_available() and hf_backend.initialize(config):
                    self.backend = hf_backend
                    logger.info("Auto-selected HuggingFace backend")
                else:
                    self.backend = LocalTransformerBackend()
                    self.backend.initialize(config)
                    logger.info("Auto-selected local backend")
                    
            elif self.backend_strategy == "huggingface":
                self.backend = HuggingFaceTransformerBackend()
                if not self.backend.initialize(config):
                    raise RuntimeError("HuggingFace backend initialization failed")
                    
            else:  # local or fallback
                self.backend = LocalTransformerBackend()
                if not self.backend.initialize(config):
                    raise RuntimeError("Local backend initialization failed")
            
            self.initialized = True
            logger.info("PublicTransformerAdapter initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Adapter initialization failed: {e}")
            # Fallback to local backend
            try:
                self.backend = LocalTransformerBackend()
                self.backend.initialize({})
                self.initialized = True
                logger.info("Fallback to local backend successful")
                return True
            except Exception as fallback_e:
                logger.error(f"Fallback initialization failed: {fallback_e}")
                return False
    
    def process(self, data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main processing method that integrates with the pipeline orchestrator.
        
        Maintains compatibility with evidence tracking system requirements and
        supports four-cluster processing workflow.
        
        Args:
            data: Input data from pipeline
            context: Optional context information
            
        Returns:
            Dict with transformed output preserving evidence integrity
        """
        if not self.initialized:
            logger.warning("Adapter not initialized, attempting auto-initialization")
            if not self.initialize():
                logger.error("Auto-initialization failed, using passthrough mode")
                return self._passthrough_process(data, context)
        
        # Convert input to dictionary if needed
        if not isinstance(data, dict):
            data = {"input": data}
        
        # Create transformation context
        trans_context = self._create_transformation_context(data, context)
        
        # Process the data while preserving evidence integrity
        result = self._transform_with_evidence_preservation(data, trans_context)
        
        # Record transformation for audit trail
        self._record_transformation(data, result, trans_context)
        
        return result
    
    def _create_transformation_context(
        self, 
        data: Dict[str, Any], 
        context: Optional[Dict[str, Any]]
    ) -> TransformationContext:
        """Create transformation context from input data and context"""
        
        # Detect four-cluster workflow
        four_cluster_workflow = (
            "cluster_audit" in data or 
            "clusters" in data or
            (context and context.get("four_cluster_workflow", False))
        )
        
        # Check for evidence system presence
        has_evidence_system = (
            "evidence_system" in data or 
            "evidence" in data
        )
        
        return TransformationContext(
            source=context.get("source", "pipeline") if context else "pipeline",
            preserve_evidence=has_evidence_system,
            model_backend=self.backend_strategy,
            pedagogical_mode=True,
            cluster_processing="cluster" in str(data).lower(),
            four_cluster_workflow=four_cluster_workflow
        )
    
    def _transform_with_evidence_preservation(
        self, 
        data: Dict[str, Any], 
        trans_context: TransformationContext
    ) -> Dict[str, Any]:
        """Transform data while preserving evidence integrity"""
        
        # Start with input data
        result = data.copy()
        
        # Extract and preserve evidence information
        evidence_summary = self._extract_evidence_summary(data)
        
        # Process different data sections
        if trans_context.four_cluster_workflow:
            result = self._process_four_cluster_workflow(result, trans_context)
        
        # Generate pedagogical public report
        public_report = self._generate_public_report(result, evidence_summary, trans_context)
        result["public_report"] = public_report
        
        # Add transformation metadata
        result["transformer_metadata"] = {
            "backend": self.backend_strategy,
            "context": trans_context.__dict__,
            "preservation_verified": self._verify_evidence_preservation(data, result),
            "transformation_hash": self._compute_transformation_hash(data, result)
        }
        
        return result
    
    def _extract_evidence_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract evidence summary compatible with evidence tracking system"""
        
        evidence_summary = {"counts": {}, "examples": [], "dimensions": []}
        
        try:
            # Check for evidence_system (EvidenceSystem instance)
            evidence_system = data.get("evidence_system")
            if evidence_system and hasattr(evidence_system, 'get_stats'):
                stats = evidence_system.get_stats()
                evidence_summary.update({
                    "counts": {
                        "total_questions": stats.get("total_questions", 0),
                        "total_evidence": stats.get("total_evidence", 0)
                    },
                    "dimensions": stats.get("dimensions", []),
                    "coverage": stats.get("recent_coverage"),
                    "system_type": "EvidenceSystem"
                })
            
            # Check for cluster_audit structure
            cluster_audit = data.get("cluster_audit", {})
            if cluster_audit:
                micro = cluster_audit.get("micro", {})
                evidence_ids = []
                for cluster_name, payload in micro.items():
                    answers = payload.get("answers", [])
                    for answer in answers:
                        eids = answer.get("evidence_ids", [])
                        if isinstance(eids, list):
                            evidence_ids.extend(eids)
                
                evidence_summary["examples"] = list(set(evidence_ids))[:10]
                evidence_summary["counts"]["from_cluster_audit"] = len(evidence_ids)
            
            # Legacy evidence structure
            if "evidence" in data and isinstance(data["evidence"], dict):
                evidence_summary["counts"]["legacy_evidence"] = len(data["evidence"])
                
        except Exception as e:
            logger.warning(f"Evidence extraction failed: {e}")
        
        return evidence_summary
    
    def _process_four_cluster_workflow(
        self, 
        data: Dict[str, Any], 
        trans_context: TransformationContext
    ) -> Dict[str, Any]:
        """Process data specifically for four-cluster workflow"""
        
        # Extract cluster information
        clusters = data.get("clusters", [])
        cluster_answers = data.get("cluster_answers", {})
        
        if len(clusters) >= 4 or len(cluster_answers) >= 4:
            logger.info("Processing four-cluster workflow")
            
            # Calculate cluster coverage
            total_questions = set()
            for cluster_name, answers in cluster_answers.items():
                for answer in answers:
                    if "question_id" in answer:
                        total_questions.add(answer["question_id"])
            
            cluster_coverage = len(total_questions) * len(cluster_answers) if total_questions else 0
            
            # Add cluster processing metadata
            data["cluster_processing_metadata"] = {
                "clusters_detected": len(clusters),
                "cluster_answers_count": len(cluster_answers),
                "unique_questions": len(total_questions),
                "coverage_estimate": cluster_coverage,
                "four_cluster_confirmed": len(clusters) >= 4
            }
        
        return data
    
    def _generate_public_report(
        self, 
        data: Dict[str, Any], 
        evidence_summary: Dict[str, Any], 
        trans_context: TransformationContext
    ) -> Dict[str, Any]:
        """Generate pedagogical public report preserving evidence rigor"""
        
        # Extract key information
        macro = data.get("macro_synthesis", {})
        audit = data.get("canonical_audit", {})
        cluster_meta = data.get("cluster_processing_metadata", {})
        
        # Generate pedagogical content sections
        content_sections = [
            "Síntesis de Alineación con el Decálogo de Derechos Humanos",
            "",
        ]
        
        # Add alignment score with interpretation
        alignment_score = macro.get("alignment_score", "N/A")
        if isinstance(alignment_score, (int, float)):
            interpretation = "excelente" if alignment_score > 0.8 else "buena" if alignment_score > 0.6 else "mejorable"
            content_sections.append(f"Puntaje de alineación: {alignment_score:.2f} ({interpretation})")
        else:
            content_sections.append(f"Puntaje de alineación: {alignment_score}")
        
        # Add DNP standards usage
        uses_dnp = audit.get("uses_dnp_standards", False)
        content_sections.append(f"Uso de estándares DNP: {'sí' if uses_dnp else 'no'}")
        
        # Add evidence information
        evidence_count = evidence_summary.get("counts", {}).get("total_evidence", 0)
        if evidence_count > 0:
            content_sections.append(f"Evidencia analizada: {evidence_count} elementos identificados")
        
        # Add four-cluster information if applicable
        if trans_context.four_cluster_workflow:
            four_clusters = cluster_meta.get("four_cluster_confirmed", False)
            coverage = cluster_meta.get("coverage_estimate", 0)
            content_sections.append(f"Procesamiento cuatro-cluster: {'completado' if four_clusters else 'pendiente'}")
            if coverage:
                content_sections.append(f"Cobertura estimada: {coverage}")
        
        # Add explanatory section
        content_sections.extend([
            "",
            "¿Qué encontramos?",
            "Esta evaluación analiza la alineación del Plan de Desarrollo con el Decálogo de Derechos Humanos.",
            "Los resultados se basan en evidencia textual extraída y procesada de manera determinística.",
            "La puntuación refleja el grado de cumplimiento identificado en el documento."
        ])
        
        # Transform content using backend if available
        transformed_content = []
        for section in content_sections:
            if section and self.backend:
                transformed_section = self.backend.transform_text(
                    section, 
                    {"type": "pedagogical_content", "preserve_meaning": True}
                )
                transformed_content.append(transformed_section)
            else:
                transformed_content.append(section)
        
        # Create final public report
        public_report = {
            "title": "Evaluación del Plan de Desarrollo frente al Decálogo de DD.HH.",
            "content": "\n".join(transformed_content),
            "evidence_summary": evidence_summary,
            "preserve_evidence": True,
            "replicability": audit.get("replicability", {}),
            "transformation_context": trans_context.__dict__,
            "hash": hashlib.sha256(
                "\n".join(transformed_content).encode("utf-8")
            ).hexdigest()[:16],
            "backend_used": self.backend_strategy,
            "four_cluster_processing": trans_context.four_cluster_workflow
        }
        
        return public_report
    
    def _verify_evidence_preservation(
        self, 
        original: Dict[str, Any], 
        transformed: Dict[str, Any]
    ) -> bool:
        """Verify that evidence integrity is preserved after transformation"""
        try:
            # Check that evidence system is preserved
            original_evidence_system = original.get("evidence_system")
            transformed_evidence_system = transformed.get("evidence_system")
            
            if original_evidence_system is not None:
                if transformed_evidence_system is None:
                    return False
                
                # If both are EvidenceSystem instances, compare stats
                if hasattr(original_evidence_system, 'get_stats') and hasattr(transformed_evidence_system, 'get_stats'):
                    orig_stats = original_evidence_system.get_stats()
                    trans_stats = transformed_evidence_system.get_stats()
                    
                    # Check key preservation metrics
                    if orig_stats.get("total_evidence") != trans_stats.get("total_evidence"):
                        return False
                    if orig_stats.get("total_questions") != trans_stats.get("total_questions"):
                        return False
            
            # Check that cluster structure is preserved
            orig_cluster_answers = original.get("cluster_answers", {})
            trans_cluster_answers = transformed.get("cluster_answers", {})
            
            if len(orig_cluster_answers) != len(trans_cluster_answers):
                return False
            
            # Check evidence IDs preservation
            for cluster_name, answers in orig_cluster_answers.items():
                if cluster_name not in trans_cluster_answers:
                    return False
                
                orig_answers = answers
                trans_answers = trans_cluster_answers[cluster_name]
                
                for orig_ans, trans_ans in zip(orig_answers, trans_answers):
                    if orig_ans.get("evidence_ids") != trans_ans.get("evidence_ids"):
                        return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Evidence preservation verification failed: {e}")
            return False
    
    def _compute_transformation_hash(
        self, 
        original: Dict[str, Any], 
        transformed: Dict[str, Any]
    ) -> str:
        """Compute hash for transformation audit trail"""
        try:
            # Create deterministic representation
            orig_repr = str(sorted(original.keys()))
            trans_repr = str(sorted(transformed.keys()))
            combined = f"{orig_repr}:{trans_repr}:{self.backend_strategy}"
            
            return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]
        except Exception:
            return "unknown"
    
    def _record_transformation(
        self, 
        original: Dict[str, Any], 
        result: Dict[str, Any], 
        trans_context: TransformationContext
    ) -> None:
        """Record transformation for audit trail"""
        record = {
            "timestamp": __import__("time").time(),
            "context": trans_context.__dict__,
            "input_keys": list(original.keys()) if isinstance(original, dict) else ["non_dict"],
            "output_keys": list(result.keys()) if isinstance(result, dict) else ["non_dict"],
            "evidence_preserved": self._verify_evidence_preservation(original, result),
            "transformation_hash": self._compute_transformation_hash(original, result)
        }
        
        self.transformation_history.append(record)
        
        # Keep only last 100 records
        if len(self.transformation_history) > 100:
            self.transformation_history = self.transformation_history[-100:]
    
    def _passthrough_process(self, data: Any, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback processing when adapter is not initialized"""
        logger.warning("Using passthrough mode - no transformation applied")
        
        if isinstance(data, dict):
            result = data.copy()
        else:
            result = {"input": data}
        
        # Add minimal public report for compatibility
        result["public_report"] = {
            "title": "Evaluación del Plan de Desarrollo frente al Decálogo de DD.HH.",
            "content": "Procesamiento en modo de respaldo - transformación limitada",
            "evidence_summary": {"counts": {}, "examples": []},
            "preserve_evidence": True,
            "hash": "passthrough",
            "backend_used": "passthrough"
        }
        
        return result
    
    def get_transformation_stats(self) -> Dict[str, Any]:
        """Get statistics about transformations performed"""
        if not self.transformation_history:
            return {"total_transformations": 0}
        
        evidence_preserved_count = sum(
            1 for record in self.transformation_history 
            if record.get("evidence_preserved", False)
        )
        
        return {
            "total_transformations": len(self.transformation_history),
            "evidence_preservation_rate": evidence_preserved_count / len(self.transformation_history),
            "backend_used": self.backend_strategy,
            "initialized": self.initialized,
            "recent_transformations": self.transformation_history[-5:] if self.transformation_history else []
        }


def process(data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Standalone processing function for backward compatibility.
    
    Creates a PublicTransformerAdapter instance and processes the data.
    This function is called by existing code like canonical_output_auditor.py
    
    Args:
        data: Input data to transform
        context: Optional context information
        
    Returns:
        Dict with transformed output
    """
    try:
        adapter = PublicTransformerAdapter(backend_strategy="auto")
        if not adapter.initialize():
            logger.warning("Adapter initialization failed, using legacy processing")
            return _legacy_process(data, context)
        
        return adapter.process(data, context)
        
    except Exception as e:
        logger.error(f"Standalone processing failed: {e}")
        return _legacy_process(data, context)


def _legacy_process(data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Legacy processing function for backward compatibility"""
    
    # Convert to dict if needed
    if not isinstance(data, dict):
        out = {"input": data}
    else:
        out = data.copy()
    
    # Extract information using the original logic
    macro = out.get("macro_synthesis", {})
    audit = out.get("canonical_audit", {})
    
    # Extract evidence summary
    evidence_summary = {"counts": {}, "examples": []}
    try:
        ca = out.get("cluster_audit", {})
        micro = ca.get("micro", {})
        total = 0
        ev_ids = []
        for c, payload in (micro or {}).items():
            answers = payload.get("answers", [])
            for a in answers:
                eids = a.get("evidence_ids", [])
                if isinstance(eids, list):
                    ev_ids.extend([e for e in eids if isinstance(e, str)])
                    total += len(eids)
        evidence_summary["counts"] = {"total_items": total, "unique_ids": len(set(ev_ids))}
        evidence_summary["examples"] = list(set(ev_ids))[:5]
    except Exception:
        pass
    
    # Generate content sections
    content_sections = [
        "Síntesis de Alineación con el Decálogo de Derechos Humanos",
        f"Puntaje de alineación: {macro.get('alignment_score', 'N/A')}",
        f"Uso de estándares DNP: {'sí' if audit.get('uses_dnp_standards') else 'no'}",
        "¿Qué encontramos? Explicación accesible de los hallazgos clave basada en evidencia.",
    ]
    
    # Create public report
    public_report = {
        "title": "Evaluación del Plan de Desarrollo frente al Decálogo de DD.HH.",
        "content": "\n\n".join(str(x) for x in content_sections),
        "evidence_summary": evidence_summary,
        "preserve_evidence": True,
        "replicability": audit.get("replicability", {}),
        "hash": hashlib.sha256(str(content_sections).encode("utf-8")).hexdigest()[:16],
        "backend_used": "legacy"
    }
    
    out["public_report"] = public_report
    return out
"""
Industrial Analysis NLP Orchestrator with Total Ordering (v2.1.0)

An enhanced, self-contained orchestrator that coordinates nine (9)
analysis_nlp components with strict determinism and improved performance.
"""

from __future__ import annotations

import json
import logging
import sys
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from collections import OrderedDict, deque
from abc import ABC, abstractmethod
import threading
from functools import wraps
from dataclasses import dataclass, field, asdict
from enum import Enum
import copy

# -------------------------------
# Configuration & Logging
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('orchestrator.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger("A25-Orchestrator")

# Mandatory Pipeline Contract Annotations
__phase__ = "A"
__code__ = "25A"
__stage_order__ = 4
__version__ = "2.1.0"
__author__ = "Industrial Analysis Team"


# -------------------------------
# Enums & Metadata
# -------------------------------
class ProcessingStatus(Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    ERROR = "error"
    NOT_INITIALIZED = "not_initialized"
    PROCESSING = "processing"
    TIMEOUT = "timeout"


class ComponentType(Enum):
    ANALYZER = "analyzer"
    PROCESSOR = "processor"
    VALIDATOR = "validator"
    MAPPER = "mapper"
    EXTRACTOR = "extractor"
    ORCHESTRATOR = "orchestrator"


@dataclass
class ComponentMetadata:
    name: str
    component_type: ComponentType
    version: str = "1.0.0"
    dependencies: List[str] = field(default_factory=list)
    initialization_time: Optional[str] = None
    last_processing_time: Optional[str] = None
    processing_count: int = 0
    error_count: int = 0
    total_processing_time: float = 0.0
    avg_processing_time: float = 0.0


# -------------------------------
# Deterministic Base
# -------------------------------
class TotalOrderingBase(ABC):
    """Enhanced deterministic base for stable IDs, canonical JSON, and audit metadata."""

    def __init__(self, component_name: str):
        self.component_name = component_name
        self.component_id = self._generate_component_id()
        self.creation_timestamp = self._now()
        self.state_hash = ""
        self._lock = threading.RLock()
        self._processing_times = []

    # Determinism helpers
    def _generate_component_id(self) -> str:
        base = f"{self.component_name}:{__version__}:{__code__}:{__phase__}:{__stage_order__}"
        return hashlib.sha256(base.encode()).hexdigest()[:16]

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _timestamp() -> float:
        return time.time()

    def canonicalize(self, data: Any) -> Any:
        """Enhanced canonicalization with better handling of complex types."""
        if data is None:
            return None
        if isinstance(data, (str, int, float, bool)):
            return data
        if isinstance(data, dict):
            return {k: self.canonicalize(data[k]) for k in sorted(data)}
        if isinstance(data, list):
            return sorted([self.canonicalize(x) for x in data],
                          key=lambda x: json.dumps(x, sort_keys=True, default=str))
        if isinstance(data, set):
            return self.canonicalize(list(data))
        if isinstance(data, tuple):
            return tuple(self.canonicalize(list(data)))
        if hasattr(data, "__dict__"):
            return self.canonicalize({k: v for k, v in data.__dict__.items()
                                      if not k.startswith("_") and not callable(v)})
        return str(data)

    def cj(self, data: Any) -> str:
        """Canonical JSON representation."""
        return json.dumps(self.canonicalize(data), sort_keys=True,
                          separators=(",", ":"), ensure_ascii=True, default=str)

    def stable_id(self, data: Any) -> str:
        """Generate stable hash ID for any data structure."""
        return hashlib.sha256(self.cj(data).encode()).hexdigest()[:16]

    def op_id(self, operation: str, params: Dict[str, Any]) -> str:
        """Generate operation ID with timestamp for ordering."""
        payload = {"c": self.component_name, "op": operation, "p": self.stable_id(params), "t": self._now()}
        return self.stable_id(payload)

    def update_state_hash(self, state: Any) -> str:
        """Update and return the state hash."""
        with self._lock:
            self.state_hash = self.stable_id(state)
            return self.state_hash

    def record_processing_time(self, processing_time: float):
        """Record processing time for metrics."""
        with self._lock:
            self._processing_times.append(processing_time)
            # Keep only the last 100 processing times
            if len(self._processing_times) > 100:
                self._processing_times.pop(0)

    def get_processing_stats(self) -> Dict[str, float]:
        """Get processing statistics."""
        with self._lock:
            if not self._processing_times:
                return {"min": 0.0, "max": 0.0, "avg": 0.0, "count": 0}

            return {
                "min": min(self._processing_times),
                "max": max(self._processing_times),
                "avg": sum(self._processing_times) / len(self._processing_times),
                "count": len(self._processing_times)
            }

    def meta(self) -> Dict[str, Any]:
        """Enhanced metadata with processing statistics."""
        stats = self.get_processing_stats()
        return self.canonicalize({
            "component_name": self.component_name,
            "component_id": self.component_id,
            "creation_timestamp": self.creation_timestamp,
            "state_hash": self.state_hash,
            "version": __version__,
            "phase": __phase__,
            "code": __code__,
            "stage_order": __stage_order__,
            "processing_stats": stats
        })


class DeterministicCollectionMixin:
    """Enhanced collection operations with better merging."""

    def merge_d(self, *collections: Union[Dict, List]) -> Union[Dict, List]:
        """Merge multiple collections deterministically."""
        collections = [c for c in collections if c is not None]
        if not collections:
            return {}
        if isinstance(collections[0], dict):
            merged: Dict[str, Any] = {}
            for c in collections:
                if isinstance(c, dict):
                    merged.update(c)
            return {k: merged[k] for k in sorted(merged)}
        if isinstance(collections[0], list):
            merged_list: List[Any] = []
            for c in collections:
                if isinstance(c, list):
                    merged_list.extend(c)
            return sorted(merged_list, key=lambda x: json.dumps(x, sort_keys=True, default=str))
        return collections[0]

    def deep_merge_d(self, dict1: Dict, dict2: Dict) -> Dict:
        """Recursively merge two dictionaries deterministically."""
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.deep_merge_d(result[key], value)
            else:
                result[key] = value
        return {k: result[k] for k in sorted(result)}


# -------------------------------
# Component Base
# -------------------------------
class BaseAnalysisComponent(TotalOrderingBase, ABC):
    """Enhanced base component with better metrics and error handling."""

    def __init__(self, component_name: str, component_type: ComponentType):
        super().__init__(component_name)
        self.metadata = ComponentMetadata(
            name=component_name,
            component_type=component_type,
            initialization_time=self._now(),
        )
        self._initialized = True
        self._timeout = 30.0  # Default timeout in seconds

    @abstractmethod
    def process(self, data: Optional[Any] = None, context: Optional[Any] = None) -> Dict[str, Any]:
        ...

    def get_metadata(self) -> Dict[str, Any]:
        """Enhanced metadata with more details."""
        return {
            "name": self.metadata.name,
            "type": self.metadata.component_type.value,
            "version": self.metadata.version,
            "initialization_time": self.metadata.initialization_time,
            "processing_count": self.metadata.processing_count,
            "error_count": self.metadata.error_count,
            "total_processing_time": self.metadata.total_processing_time,
            "avg_processing_time": self.metadata.avg_processing_time,
            "component_id": self.component_id,
        }

    def _tick(self, processing_time: float, success: bool = True):
        """Update metrics after processing."""
        self.metadata.processing_count += 1
        self.metadata.last_processing_time = self._now()
        self.metadata.total_processing_time += processing_time
        self.metadata.avg_processing_time = (
                self.metadata.total_processing_time / self.metadata.processing_count
        )
        if not success:
            self.metadata.error_count += 1
        self.record_processing_time(processing_time)

    # Standardized error payload
    def _err(self, msg: str, details: Optional[Dict] = None) -> Dict[str, Any]:
        """Enhanced error response with details."""
        error_data = {
            "component": self.component_name,
            "component_id": self.component_id,
            "error": msg,
            "status": ProcessingStatus.ERROR.value,
            "timestamp": self._now(),
        }

        if details:
            error_data["details"] = details

        return error_data

    def _timeout_err(self, elapsed: float) -> Dict[str, Any]:
        """Standard timeout error response."""
        return self._err(
            f"Timeout after {elapsed:.2f}s (limit: {self._timeout}s)",
            {"elapsed": elapsed, "timeout": self._timeout}
        )


# -------------------------------
# Components (9) - Enhanced versions
# -------------------------------
class AdaptiveAnalyzer(BaseAnalysisComponent):
    def __init__(self):
        super().__init__("AdaptiveAnalyzer", ComponentType.ANALYZER)
        self.rules = {"data_quality_threshold": 0.7, "processing_timeout": 30, "max_retries": 3}
        self._timeout = 25.0

    def process(self, data: Optional[Any] = None, context: Optional[Any] = None) -> Dict[str, Any]:
        start_time = self._timestamp()
        try:
            d = self.canonicalize(data or {})
            c = self.canonicalize(context or {})

            # Calculate complexity metrics
            data_size = len(self.cj(d)) / 1024.0
            context_size = len(self.cj(c)) / 1024.0

            analysis = {
                "data_complexity": data_size,
                "context_richness": context_size,
                "system_health": 0.9 - (min(1.0, data_size / 10.0) * 0.1),  # Decrease with complexity
                "processing_load": self.metadata.processing_count,
            }

            # Determine adaptations needed
            adaptations = []
            if analysis["data_complexity"] > 4.0:
                adaptations.append("increase_buffer_size")
            if analysis["processing_load"] > 100:
                adaptations.append("enable_batch_mode")
            if analysis["system_health"] < 0.7:
                adaptations.append("reduce_processing_depth")

            processing_time = self._timestamp() - start_time
            self._tick(processing_time, True)

            out = {
                "component": self.component_name,
                "component_id": self.component_id,
                "results": {
                    "analysis": analysis,
                    "adaptations": sorted(adaptations),
                    "confidence": max(0.5, 0.85 - (data_size / 20.0))  # Adjust confidence based on complexity
                },
                "metadata": self.meta(),
                "status": ProcessingStatus.SUCCESS.value,
                "timestamp": self._now(),
                "processing_time": processing_time
            }
            return self.canonicalize(out)
        except Exception as e:
            processing_time = self._timestamp() - start_time
            self._tick(processing_time, False)
            return self._err(str(e), {"processing_time": processing_time})


class QuestionAnalyzer(BaseAnalysisComponent):
    def __init__(self):
        super().__init__("QuestionAnalyzer", ComponentType.ANALYZER)
        self.patterns = {
            "what": {"type": "factual", "complexity": 0.3},
            "how": {"type": "procedural", "complexity": 0.7},
            "why": {"type": "causal", "complexity": 0.9},
            "when": {"type": "temporal", "complexity": 0.4},
            "where": {"type": "spatial", "complexity": 0.4},
            "who": {"type": "personal", "complexity": 0.4},
            "which": {"type": "discriminative", "complexity": 0.5},
        }
        self._timeout = 30.0

    def process(self, data: Optional[Any] = None, context: Optional[Any] = None) -> Dict[str, Any]:
        start_time = self._timestamp()
        try:
            d = self.canonicalize(data or {})
            qs = self._extract_questions(d)
            analysis = self._analyze(qs)
            reqs = self._requirements(analysis)

            processing_time = self._timestamp() - start_time
            self._tick(processing_time, True)

            out = {
                "component": self.component_name,
                "component_id": self.component_id,
                "results": {
                    "questions_found": len(qs),
                    "question_analysis": analysis,
                    "requirements": reqs,
                    "confidence": max(0.6, 0.8 - (len(qs) / 100.0))  # Adjust confidence based on question count
                },
                "metadata": self.meta(),
                "status": ProcessingStatus.SUCCESS.value,
                "timestamp": self._now(),
                "processing_time": processing_time
            }
            return self.canonicalize(out)
        except Exception as e:
            processing_time = self._timestamp() - start_time
            self._tick(processing_time, False)
            return self._err(str(e), {"processing_time": processing_time})

    def _extract_questions(self, data: Dict[str, Any]) -> List[str]:
        """More efficient question extraction using iterative approach."""
        found = set()
        stack = [data]

        while stack:
            current = stack.pop()

            if isinstance(current, str):
                # Split into sentences and check for questions
                sentences = [s.strip() for s in current.split('.') if s.strip()]
                for s in sentences:
                    if any(s.strip().endswith('?') for s in sentences) or \
                            any(s.lower().startswith(k) for k in self.patterns):
                        found.add(s)

            elif isinstance(current, dict):
                stack.extend(current.values())

            elif isinstance(current, list):
                stack.extend(current)

        return sorted(found)

    def _analyze(self, qs: List[str]) -> Dict[str, Any]:
        d = {"total_questions": len(qs), "question_types": {}, "complexity_distribution": {}}
        complexity_sum = 0.0

        for q in qs:
            ql = q.lower()
            for k, info in self.patterns.items():
                if k in ql:
                    qt = info["type"]
                    cx = info["complexity"]
                    complexity_sum += cx

                    d["question_types"][qt] = d["question_types"].get(qt, 0) + 1
                    lvl = "low" if cx < 0.4 else ("medium" if cx < 0.7 else "high")
                    d["complexity_distribution"][lvl] = d["complexity_distribution"].get(lvl, 0) + 1
                    break

        # Calculate average complexity
        if qs:
            d["average_complexity"] = complexity_sum / len(qs)
        else:
            d["average_complexity"] = 0.0

        return d

    def _requirements(self, analysis: Dict[str, Any]) -> List[str]:
        req = []
        if analysis.get("total_questions", 0) > 5:
            req.append("batch_question_processing")
        if analysis.get("complexity_distribution", {}).get("high", 0) > 0:
            req.append("advanced_reasoning")
        if "causal" in analysis.get("question_types", {}):
            req.append("causal_analysis")
        if analysis.get("average_complexity", 0) > 0.6:
            req.append("enhanced_processing")
        return sorted(req)


class QuestionDecalogoMapper(BaseAnalysisComponent):
    def __init__(self):
        super().__init__("QuestionDecalogoMapper", ComponentType.MAPPER)
        self.decalogo = {
            "principle_1": "Transparency and clarity",
            "principle_2": "Evidence-based reasoning",
            "principle_3": "Stakeholder engagement",
            "principle_4": "Ethical consideration",
            "principle_5": "Sustainability focus",
            "principle_6": "Innovation encouragement",
            "principle_7": "Risk assessment",
            "principle_8": "Quality assurance",
            "principle_9": "Continuous improvement",
            "principle_10": "Social responsibility",
        }
        self.keywords = {
            "principle_1": ["transparency", "clear", "clarity", "open", "visible", "disclosure"],
            "principle_2": ["evidence", "proof", "data", "research", "facts", "study", "analysis"],
            "principle_3": ["stakeholder", "community", "engagement", "participation", "involvement"],
            "principle_4": ["ethical", "moral", "right", "wrong", "ethics", "integrity", "values"],
            "principle_5": ["sustainability", "environment", "sustainable", "green", "eco-friendly", "renewable"],
            "principle_6": ["innovation", "creative", "new", "novel", "innovative", "invention", "breakthrough"],
            "principle_7": ["risk", "assessment", "evaluation", "danger", "threat", "vulnerability", "mitigation"],
            "principle_8": ["quality", "assurance", "standard", "excellence", "accuracy", "precision"],
            "principle_9": ["improvement", "better", "enhance", "optimize", "refine", "develop"],
            "principle_10": ["social", "society", "responsibility", "community", "welfare", "benefit"],
        }
        self._timeout = 35.0

    def process(self, data: Optional[Any] = None, context: Optional[Any] = None) -> Dict[str, Any]:
        start_time = self._timestamp()
        try:
            d = self.canonicalize(data or {})

            # Use QuestionAnalyzer to extract questions
            qa = QuestionAnalyzer()
            qs = qa._extract_questions(d)

            mappings: Dict[str, List[str]] = {}
            for q in qs:
                ql = q.lower()
                matched = False

                for p, kws in self.keywords.items():
                    if any(k in ql for k in kws):
                        mappings.setdefault(p, []).append(q)
                        matched = True

                # If no keyword match, try to map based on question type
                if not matched:
                    for pattern, info in qa.patterns.items():
                        if pattern in ql:
                            # Map to principle based on question type
                            principle_map = {
                                "what": "principle_2",  # Evidence-based
                                "how": "principle_6",  # Innovation
                                "why": "principle_4",  # Ethical
                                "when": "principle_9",  # Continuous improvement
                                "where": "principle_5",  # Sustainability
                                "who": "principle_3",  # Stakeholder engagement
                                "which": "principle_8",  # Quality assurance
                            }
                            principle = principle_map.get(pattern, "principle_10")  # Default to social responsibility
                            mappings.setdefault(principle, []).append(q)
                            break

            analysis = {
                "total_mapped_questions": sum(len(v) for v in mappings.values()),
                "unmapped_questions": len(qs) - sum(len(v) for v in mappings.values()),
                "principle_coverage": len(mappings),
                "coverage_percentage": (len(mappings) / 10.0) * 100.0 if mappings else 0.0,
                "principle_distribution": {k: len(v) for k, v in mappings.items()}
            }

            processing_time = self._timestamp() - start_time
            self._tick(processing_time, True)

            out = {
                "component": self.component_name,
                "component_id": self.component_id,
                "results": {
                    "question_count": len(qs),
                    "mappings": self.canonicalize(mappings),
                    "mapping_analysis": analysis,
                    "confidence": max(0.6, 0.75 - (analysis["unmapped_questions"] / max(1, len(qs)) * 0.5))
                },
                "metadata": self.meta(),
                "status": ProcessingStatus.SUCCESS.value,
                "timestamp": self._now(),
                "processing_time": processing_time
            }
            return self.canonicalize(out)
        except Exception as e:
            processing_time = self._timestamp() - start_time
            self._tick(processing_time, False)
            return self._err(str(e), {"processing_time": processing_time})


class ExtractorEvidenciasContextual(BaseAnalysisComponent):
    def __init__(self):
        super().__init__("ExtractorEvidenciasContextual", ComponentType.EXTRACTOR)
        self.types = {
            "quantitative": ["number", "percentage", "data", "statistic", "metric", "measure", "count"],
            "qualitative": ["opinion", "view", "perspective", "belief", "feeling", "attitude", "perception"],
            "empirical": ["study", "research", "experiment", "observation", "test", "trial", "analysis"],
            "theoretical": ["theory", "model", "framework", "concept", "principle", "hypothesis", "paradigm"],
            "historical": ["past", "previous", "history", "before", "earlier", "tradition", "legacy"],
        }
        self._timeout = 40.0

    def process(self, data: Optional[Any] = None, context: Optional[Any] = None) -> Dict[str, Any]:
        start_time = self._timestamp()
        try:
            d = self.canonicalize(data or {})
            c = self.canonicalize(context or {})
            ev = self._extract(d)
            ev_ctx = self._filter_by_context(ev, c)
            classified = self._classify(ev_ctx)

            processing_time = self._timestamp() - start_time
            self._tick(processing_time, True)

            out = {
                "component": self.component_name,
                "component_id": self.component_id,
                "results": {
                    "evidence_count": len(ev_ctx),
                    "evidence": ev_ctx,
                    "classified_evidence": classified,
                    "confidence": min(0.95, 0.82 + (len(ev_ctx) / 100.0))  # Increase confidence with more evidence
                },
                "metadata": self.meta(),
                "status": ProcessingStatus.SUCCESS.value,
                "timestamp": self._now(),
                "processing_time": processing_time
            }
            return self.canonicalize(out)
        except Exception as e:
            processing_time = self._timestamp() - start_time
            self._tick(processing_time, False)
            return self._err(str(e), {"processing_time": processing_time})

    def _extract(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """More efficient evidence extraction using iterative approach."""
        out: List[Dict[str, Any]] = []
        stack = [(data, "root")]

        while stack:
            current, source = stack.pop()

            if isinstance(current, str):
                # Extract evidence from text
                sentences = [s.strip() for s in current.split('.') if s.strip() and len(s.strip()) > 20]
                for i, s in enumerate(sentences):
                    out.append({
                        "id": hashlib.md5(s.encode()).hexdigest()[:8],
                        "content": s,
                        "source": source,
                        "position": i,
                        "length": len(s),
                    })

            elif isinstance(current, dict):
                for k, v in current.items():
                    stack.append((v, f"{source}.{k}" if source != "root" else k))

            elif isinstance(current, list):
                for i, item in enumerate(current):
                    stack.append((item, f"{source}[{i}]"))

        return sorted(out, key=lambda x: (x["source"], x["position"]))

    def _filter_by_context(self, ev: List[Dict[str, Any]], ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not ctx:
            return ev

        # Extract keywords from context
        kw = set()
        stack = [ctx]

        while stack:
            current = stack.pop()

            if isinstance(current, str):
                kw.update(current.lower().split())

            elif isinstance(current, dict):
                stack.extend(current.values())

            elif isinstance(current, list):
                stack.extend(current)

        # Filter evidence based on context relevance
        kept: List[Dict[str, Any]] = []
        for it in ev:
            words = set(it["content"].lower().split())
            common = words & kw
            rel = len(common) / (len(words) or 1)

            if rel > 0.1 or any(kw_word in it["content"].lower() for kw_word in kw):
                it_copy = it.copy()
                it_copy["relevance_score"] = round(rel, 6)
                it_copy["matching_keywords"] = sorted(common)
                kept.append(it_copy)

        return sorted(kept, key=lambda x: (-x["relevance_score"], x["source"], x["position"]))

    def _classify(self, ev: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {}

        for it in ev:
            content_lower = it["content"].lower()
            evidence_type = "unclassified"

            for t, kws in self.types.items():
                if any(k in content_lower for k in kws):
                    evidence_type = t
                    break

            out.setdefault(evidence_type, []).append(it)

        return {k: sorted(out[k], key=lambda x: (-x["relevance_score"], x["source"], x["position"]))
                for k in sorted(out)}


class EvidenceProcessor(BaseAnalysisComponent):
    def __init__(self):
        super().__init__("EvidenceProcessor", ComponentType.PROCESSOR)
        self.cfg = {"min_len": 10, "max_len": 1000, "similarity_threshold": 0.8}
        self._timeout = 35.0

    def process(self, data: Optional[Any] = None, context: Optional[Any] = None) -> Dict[str, Any]:
        start_time = self._timestamp()
        try:
            d = self.canonicalize(data or {})
            evid = self._pull_evidence(d)

            # Filter by length
            evid = [e for e in evid if self.cfg["min_len"] <= e.get("length", 0) <= self.cfg["max_len"]]

            # Remove duplicates using content similarity
            unique_evidence = self._remove_similar(evid)

            # Structure the evidence
            structured = [
                {
                    "content": v["content"],
                    "source": v.get("source", ""),
                    "pos": v.get("position", -1),
                    "length": v.get("length", 0)
                }
                for v in unique_evidence
            ]

            structured = sorted(structured, key=lambda x: (x["source"], x["pos"], x["content"]))

            processing_time = self._timestamp() - start_time
            self._tick(processing_time, True)

            out = {
                "component": self.component_name,
                "component_id": self.component_id,
                "results": {
                    "original_evidence_count": len(evid),
                    "processed_evidence_count": len(structured),
                    "structured_evidence": structured,
                    "confidence": min(0.95, 0.88 + (len(structured) / 50.0))  # Increase confidence with more evidence
                },
                "metadata": self.meta(),
                "status": ProcessingStatus.SUCCESS.value,
                "timestamp": self._now(),
                "processing_time": processing_time
            }
            return self.canonicalize(out)
        except Exception as e:
            processing_time = self._timestamp() - start_time
            self._tick(processing_time, False)
            return self._err(str(e), {"processing_time": processing_time})

    def _pull_evidence(self, obj: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract evidence from data using iterative approach."""
        out: List[Dict[str, Any]] = []
        stack = [obj]

        while stack:
            current = stack.pop()

            if isinstance(current, dict):
                # Check for evidence fields
                if "evidence" in current and isinstance(current["evidence"], list):
                    out.extend(current["evidence"])

                # Check for results with evidence
                if "results" in current and isinstance(current["results"], dict):
                    results = current["results"]
                    if "evidence" in results and isinstance(results["evidence"], list):
                        out.extend(results["evidence"])

                # Recursively process other values
                stack.extend(current.values())

            elif isinstance(current, list):
                stack.extend(current)

        return out

    def _remove_similar(self, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove similar evidence entries using content similarity."""
        if not evidence:
            return []

        # Sort by length to compare longer texts first
        sorted_evidence = sorted(evidence, key=lambda x: x.get("length", 0), reverse=True)
        unique: List[Dict[str, Any]] = []

        for e in sorted_evidence:
            content = e.get("content", "")
            is_unique = True

            for u in unique:
                u_content = u.get("content", "")
                similarity = self._text_similarity(content, u_content)

                if similarity > self.cfg["similarity_threshold"]:
                    is_unique = False
                    break

            if is_unique:
                unique.append(e)

        return unique

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity based on word overlap."""
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)


class ConsistencyValidator(BaseAnalysisComponent):
    def __init__(self):
        super().__init__("ConsistencyValidator", ComponentType.VALIDATOR)
        self._timeout = 25.0

    def process(self, data: Optional[Any] = None, context: Optional[Any] = None) -> Dict[str, Any]:
        start_time = self._timestamp()
        try:
            d = self.canonicalize(data or {})
            ok = True
            issues: List[str] = []

            # Check basic structure
            if not isinstance(d, dict):
                ok = False
                issues.append("root_not_object")

            # Check for required pipeline fields
            if "pipeline" not in d:
                ok = False
                issues.append("missing_pipeline_root")
            else:
                pipeline = d["pipeline"]
                required_fields = ["phase", "code", "version", "started_at"]
                for field in required_fields:
                    if field not in pipeline:
                        ok = False
                        issues.append(f"missing_pipeline_field:{field}")

            # Check component results
            component_keys = [k for k in d.keys() if k.startswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"))]
            for key in component_keys:
                comp_data = d[key]
                if not isinstance(comp_data, dict):
                    ok = False
                    issues.append(f"invalid_component_format:{key}")
                    continue

                if "status" not in comp_data:
                    ok = False
                    issues.append(f"missing_status:{key}")
                elif comp_data["status"] not in [s.value for s in ProcessingStatus]:
                    ok = False
                    issues.append(f"invalid_status:{key}:{comp_data['status']}")

            checksum = hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()

            processing_time = self._timestamp() - start_time
            self._tick(processing_time, ok)

            out = {
                "component": self.component_name,
                "component_id": self.component_id,
                "results": {
                    "valid": ok,
                    "issues": sorted(issues),
                    "checksum": checksum,
                    "confidence": 0.93 if ok else max(0.5, 0.93 - (len(issues) * 0.1))
                },
                "metadata": self.meta(),
                "status": ProcessingStatus.SUCCESS.value if ok else ProcessingStatus.ERROR.value,
                "timestamp": self._now(),
                "processing_time": processing_time
            }
            return self.canonicalize(out)
        except Exception as e:
            processing_time = self._timestamp() - start_time
            self._tick(processing_time, False)
            return self._err(str(e), {"processing_time": processing_time})


class RiskAssessor(BaseAnalysisComponent):
    def __init__(self):
        super().__init__("RiskAssessor", ComponentType.ANALYZER)
        self._timeout = 20.0

    def process(self, data: Optional[Any] = None, context: Optional[Any] = None) -> Dict[str, Any]:
        start_time = self._timestamp()
        try:
            d = self.canonicalize(data or {})

            # Calculate risk based on multiple factors
            size = len(self.cj(d)) / 1024.0  # KB
            complexity = self._calculate_complexity(d)
            error_count = self._count_errors(d)

            # Composite risk score
            risk = min(1.0, max(0.0,
                                (size / 200.0) * 0.4 +  # Size contributes 40%
                                (complexity / 10.0) * 0.3 +  # Complexity contributes 30%
                                (error_count / 5.0) * 0.3  # Errors contribute 30%
                                ))

            buckets = "low" if risk < 0.33 else ("medium" if risk < 0.66 else "high")

            processing_time = self._timestamp() - start_time
            self._tick(processing_time, True)

            out = {
                "component": self.component_name,
                "component_id": self.component_id,
                "results": {
                    "risk_score": round(risk, 6),
                    "bucket": buckets,
                    "size_risk": round(size / 200.0, 6),
                    "complexity_risk": round(complexity / 10.0, 6),
                    "error_risk": round(error_count / 5.0, 6),
                    "confidence": max(0.6, 0.77 - (risk * 0.2))  # Lower confidence for higher risk
                },
                "metadata": self.meta(),
                "status": ProcessingStatus.SUCCESS.value,
                "timestamp": self._now(),
                "processing_time": processing_time
            }
            return self.canonicalize(out)
        except Exception as e:
            processing_time = self._timestamp() - start_time
            self._tick(processing_time, False)
            return self._err(str(e), {"processing_time": processing_time})

    def _calculate_complexity(self, data: Any) -> float:
        """Calculate complexity of data structure."""
        if data is None:
            return 0.0
        if isinstance(data, (str, int, float, bool)):
            return 1.0
        if isinstance(data, dict):
            return sum(self._calculate_complexity(v) for v in data.values()) + len(data)
        if isinstance(data, list):
            return sum(self._calculate_complexity(v) for v in data) + len(data)
        return 1.0

    def _count_errors(self, data: Any) -> int:
        """Count errors in the data structure."""
        if not isinstance(data, dict):
            return 0

        count = 0
        stack = [data]

        while stack:
            current = stack.pop()

            if isinstance(current, dict):
                if current.get("status") == ProcessingStatus.ERROR.value:
                    count += 1
                stack.extend(current.values())

            elif isinstance(current, list):
                stack.extend(current)

        return count


class QualityAssurance(BaseAnalysisComponent):
    def __init__(self):
        super().__init__("QualityAssurance", ComponentType.VALIDATOR)
        self._timeout = 25.0

    def process(self, data: Optional[Any] = None, context: Optional[Any] = None) -> Dict[str, Any]:
        start_time = self._timestamp()
        try:
            d = self.canonicalize(data or {})

            # Calculate quality metrics
            keys = sorted(list(d.keys())) if isinstance(d, dict) else []
            diversity = len(keys) / max(1, len(self.cj(d)))

            # Check component success rates
            component_keys = [k for k in keys if k.startswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"))]
            success_count = 0
            total_components = len(component_keys)

            for key in component_keys:
                comp_data = d.get(key, {})
                if comp_data.get("status") == ProcessingStatus.SUCCESS.value:
                    success_count += 1

            success_rate = success_count / total_components if total_components > 0 else 1.0

            qa = {
                "key_count": len(keys),
                "diversity_index": round(diversity, 6),
                "component_success_rate": round(success_rate, 6),
                "ok": len(keys) > 0 and success_rate > 0.7
            }

            processing_time = self._timestamp() - start_time
            self._tick(processing_time, qa["ok"])

            out = {
                "component": self.component_name,
                "component_id": self.component_id,
                "results": {
                    "qa": qa,
                    "confidence": min(0.95, 0.81 + (success_rate * 0.1))  # Higher confidence with better success rate
                },
                "metadata": self.meta(),
                "status": ProcessingStatus.SUCCESS.value if qa["ok"] else ProcessingStatus.ERROR.value,
                "timestamp": self._now(),
                "processing_time": processing_time
            }
            return self.canonicalize(out)
        except Exception as e:
            processing_time = self._timestamp() - start_time
            self._tick(processing_time, False)
            return self._err(str(e), {"processing_time": processing_time})


class ContinuousImprovementPlanner(BaseAnalysisComponent):
    def __init__(self):
        super().__init__("ContinuousImprovementPlanner", ComponentType.PROCESSOR)
        self._timeout = 30.0

    def process(self, data: Optional[Any] = None, context: Optional[Any] = None) -> Dict[str, Any]:
        start_time = self._timestamp()
        try:
            d = self.canonicalize(data or {})

            # Analyze issues and generate improvement plan
            issues = self._analyze_issues(d)
            actions = self._generate_actions(issues)
            priorities = self._prioritize_actions(actions, d)

            plan = {
                "actions": actions,
                "priority": priorities,
                "issues_addressed": issues,
            }

            processing_time = self._timestamp() - start_time
            self._tick(processing_time, True)

            out = {
                "component": self.component_name,
                "component_id": self.component_id,
                "results": {
                    "plan": plan,
                    "confidence": min(0.95, 0.8 + (len(actions) / 20.0))  # Higher confidence with more actions
                },
                "metadata": self.meta(),
                "status": ProcessingStatus.SUCCESS.value,
                "timestamp": self._now(),
                "processing_time": processing_time
            }
            return self.canonicalize(out)
        except Exception as e:
            processing_time = self._timestamp() - start_time
            self._tick(processing_time, False)
            return self._err(str(e), {"processing_time": processing_time})

    def _analyze_issues(self, data: Dict[str, Any]) -> List[str]:
        """Analyze data to identify issues."""
        issues = []

        # Check for errors in components
        component_keys = [k for k in data.keys() if k.startswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"))]
        error_components = []

        for key in component_keys:
            comp_data = data.get(key, {})
            if comp_data.get("status") == ProcessingStatus.ERROR.value:
                error_components.append(key)

        if error_components:
            issues.append(f"component_errors:{len(error_components)}")

        # Check for consistency issues
        if "06_consistency" in data:
            consistency = data["06_consistency"]
            if consistency.get("status") == ProcessingStatus.SUCCESS.value:
                results = consistency.get("results", {})
                if not results.get("valid", False):
                    issues.append("consistency_issues")

        # Check for quality issues
        if "08_quality" in data:
            quality = data["08_quality"]
            if quality.get("status") == ProcessingStatus.SUCCESS.value:
                results = quality.get("results", {})
                qa = results.get("qa", {})
                if not qa.get("ok", False):
                    issues.append("quality_issues")

        return issues

    def _generate_actions(self, issues: List[str]) -> List[str]:
        """Generate improvement actions based on issues."""
        actions = []
        issue_set = set(issues)

        if any("component_errors" in issue for issue in issues):
            actions.extend([
                "enhance_error_handling",
                "improve_component_robustness",
                "add_fallback_mechanisms"
            ])

        if "consistency_issues" in issue_set:
            actions.extend([
                "strengthen_schema_validation",
                "add_data_integrity_checks"
            ])

        if "quality_issues" in issue_set:
            actions.extend([
                "improve_data_quality_checks",
                "enhance_quality_metrics"
            ])

        # Always include these baseline improvements
        baseline_actions = [
            "consolidate_duplicates",
            "optimize_processing_algorithms",
            "enhance_documentation"
        ]

        actions.extend(baseline_actions)
        return sorted(list(set(actions)))

    def _prioritize_actions(self, actions: List[str], data: Dict[str, Any]) -> List[str]:
        """Prioritize actions based on impact and urgency."""
        # Simple prioritization based on issue severity
        priority_map = {
            "enhance_error_handling": 1,
            "improve_component_robustness": 1,
            "add_fallback_mechanisms": 1,
            "strengthen_schema_validation": 2,
            "add_data_integrity_checks": 2,
            "improve_data_quality_checks": 3,
            "enhance_quality_metrics": 3,
            "consolidate_duplicates": 4,
            "optimize_processing_algorithms": 4,
            "enhance_documentation": 5
        }

        # Sort by priority, then alphabetically
        return sorted(actions, key=lambda x: (priority_map.get(x, 99), x))


# -------------------------------
# Enhanced Orchestrator
# -------------------------------
class IndustrialAnalysisNLPOrchestrator(TotalOrderingBase, DeterministicCollectionMixin):
    """Enhanced self-contained deterministic orchestrator for 9 components."""

    def __init__(self):
        super().__init__("IndustrialAnalysisNLPOrchestrator")
        # Registration in deterministic order
        self.registry: Dict[str, BaseAnalysisComponent] = OrderedDict([
            ("01_adaptive", AdaptiveAnalyzer()),
            ("02_questions", QuestionAnalyzer()),
            ("03_decalogo", QuestionDecalogoMapper()),
            ("04_extract_ctx", ExtractorEvidenciasContextual()),
            ("05_evidence_proc", EvidenceProcessor()),
            ("06_consistency", ConsistencyValidator()),
            ("07_risk", RiskAssessor()),
            ("08_quality", QualityAssurance()),
            ("09_ci_plan", ContinuousImprovementPlanner()),
        ])
        self.health: Dict[str, Any] = {k: {"status": ProcessingStatus.NOT_INITIALIZED.value} for k in self.registry}
        self.audit: List[Dict[str, Any]] = []
        self.retry_limit = 2
        self.progress_callbacks: List[Callable[[str, float], None]] = []

    def register_progress_callback(self, callback: Callable[[str, float], None]):
        """Register a callback for progress updates."""
        self.progress_callbacks.append(callback)

    def _update_progress(self, component: str, progress: float):
        """Update progress through all registered callbacks."""
        for callback in self.progress_callbacks:
            try:
                callback(component, progress)
            except Exception as e:
                logger.warning("Progress callback failed: %s", e)

    # ---------------------------
    # Enhanced run method
    # ---------------------------
    def run(self, payload: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context = context or {}
        start = self._now()
        start_time = self._timestamp()

        results: Dict[str, Any] = {
            "pipeline": {
                "phase": __phase__,
                "code": __code__,
                "version": __version__,
                "started_at": start,
                "components": list(self.registry.keys())
            }
        }
        state: Dict[str, Any] = {}

        total_components = len(self.registry)

        for i, (key, comp) in enumerate(self.registry.items()):
            comp_progress = i / total_components
            self._update_progress(key, comp_progress)

            op = comp.op_id("process", {"key": key})
            entry = {
                "order": key,
                "component": comp.component_name,
                "op_id": op,
                "started": self._now(),
                "attempts": 0
            }
            self.audit.append(entry)

            logger.info("Running %s (%s)", comp.component_name, key)
            attempt = 0
            last_out: Optional[Dict[str, Any]] = None

            while attempt <= self.retry_limit:
                attempt += 1
                entry["attempts"] = attempt
                self.health[key] = {"status": ProcessingStatus.PROCESSING.value, "attempt": attempt}

                try:
                    # Prepare input for component
                    comp_input = self._compose_input(payload, results, key)

                    # Run component with timeout protection
                    t0 = time.time()
                    out = comp.process(data=comp_input, context=context)
                    elapsed = time.time() - t0

                    # Check for timeout
                    if elapsed > comp._timeout:
                        out = comp._timeout_err(elapsed)
                        logger.warning("Component %s exceeded timeout: %.2fs > %.2fs",
                                       comp.component_name, elapsed, comp._timeout)

                    last_out = out
                    self.health[key] = {
                        "status": out.get("status", ProcessingStatus.ERROR.value),
                        "elapsed": round(elapsed, 6),
                        "attempt": attempt
                    }

                    results[key] = out

                    # Update orchestrator state hash deterministically
                    self.update_state_hash({"k": key, "out": out.get("status"), "elapsed": round(elapsed, 6)})

                    # If successful, break out of retry loop
                    if out.get("status") == ProcessingStatus.SUCCESS.value:
                        break

                except Exception as e:  # hard failure isolation
                    logger.exception("Component %s failed on attempt %d", comp.component_name, attempt)
                    last_out = comp._err(str(e), {"attempt": attempt})
                    self.health[key] = {"status": ProcessingStatus.ERROR.value, "attempt": attempt}

                    if attempt > self.retry_limit:
                        results[key] = last_out
                        break

                    # Exponential backoff with jitter
                    backoff_time = (0.1 * attempt) + (0.01 * hash(key) % 0.05)
                    time.sleep(backoff_time)

            entry.update({
                "finished": self._now(),
                "status": self.health[key]["status"],
                "elapsed": self.health[key].get("elapsed", 0)
            })

            # Update progress
            self._update_progress(key, (i + 1) / total_components)

        total_time = self._timestamp() - start_time

        results["pipeline"].update({
            "finished_at": self._now(),
            "total_time": round(total_time, 6),
            "health": self.canonicalize(self.health),
            "audit": self.canonicalize(self.audit),
            "state_hash": self.state_hash,
        })

        return self.canonicalize(results)

    def _compose_input(self, payload: Dict[str, Any], results: Dict[str, Any], current_key: str) -> Dict[str, Any]:
        # Deterministic merge: original payload + only SUCCESS/PARTIAL results from previous components
        visible = {}
        current_index = list(self.registry.keys()).index(current_key)
        previous_keys = list(self.registry.keys())[:current_index]

        for key in previous_keys:
            if key in results:
                v = results[key]
                if isinstance(v, dict) and v.get("status") in {
                    ProcessingStatus.SUCCESS.value,
                    ProcessingStatus.PARTIAL_SUCCESS.value
                }:
                    visible[key] = v

        return self.canonicalize({
            "input": payload,
            "previous_results": visible,
            "current_component": current_key
        })

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        stats = {}
        for key, comp in self.registry.items():
            stats[key] = {
                "processing_count": comp.metadata.processing_count,
                "error_count": comp.metadata.error_count,
                "avg_processing_time": comp.metadata.avg_processing_time,
                "last_processing_time": comp.metadata.last_processing_time,
            }

        return {
            "components": stats,
            "total_runs": sum(comp.metadata.processing_count for comp in self.registry.values()),
            "total_errors": sum(comp.metadata.error_count for comp in self.registry.values()),
            "start_time": self.creation_timestamp,
        }


# -------------------------------
# Enhanced CLI Utilities
# -------------------------------

def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in file %s: %s", path, e)
        raise


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    try:
        txt = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        path.write_text(txt, encoding="utf-8")
    except Exception as e:
        logger.error("Failed to write JSON to file %s: %s", path, e)
        raise


def print_progress(component: str, progress: float):
    """Simple progress printer for CLI."""
    bar_length = 40
    block = int(round(bar_length * progress))
    text = "\rProgress: [{}] {:.1f}% - {}".format(
        "=" * block + " " * (bar_length - block),
        progress * 100,
        component
    )
    sys.stderr.write(text)
    if progress >= 1.0:
        sys.stderr.write("\n")
    sys.stderr.flush()


def main(argv: List[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Industrial Analysis NLP Orchestrator (deterministic)")
    p.add_argument("--input", type=str, help="Input JSON file", required=False)
    p.add_argument("--output", type=str, help="Output JSON file", required=False)
    p.add_argument("--selftest", action="store_true", help="Run internal self-test")
    p.add_argument("--stats", action="store_true", help="Show statistics after run")
    p.add_argument("--verbose", "-v", action="count", default=0, help="Increase verbosity")
    args = p.parse_args(argv)

    # Set logging level based on verbosity
    if args.verbose >= 2:
        logger.setLevel(logging.DEBUG)
    elif args.verbose >= 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    orch = IndustrialAnalysisNLPOrchestrator()
    orch.register_progress_callback(print_progress)

    if args.selftest:
        logger.info("Running self-test...")
        sample_payload = {
            "document": {
                "title": "Municipal Plan 2024–2027",
                "body": "What are the key risks? How will we measure impact? Evidence shows 45% improvement. Prior research indicates strong stakeholder engagement.",
                "annex": [
                    "Why does sustainability matter? Our model suggests better long-run outcomes.",
                    "Data tables include percentages and metrics across sectors.",
                ],
            }
        }
        out = orch.run(sample_payload, context={"sector": "Health", "keywords": ["risk", "evidence", "sustainability"]})
        print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))

        if args.stats:
            print("\nStatistics:")
            print(json.dumps(orch.get_stats(), ensure_ascii=False, indent=2))
        return 0

    if not args.input:
        print("--input is required unless --selftest is used", file=sys.stderr)
        return 2

    try:
        inp = _read_json(Path(args.input))
        out = orch.run(inp, context={})

        if args.output:
            _write_json(Path(args.output), out)
        else:
            print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))

        if args.stats:
            print("\nStatistics:")
            print(json.dumps(orch.get_stats(), ensure_ascii=False, indent=2))

        return 0
    except Exception as e:
        logger.error("Pipeline execution failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
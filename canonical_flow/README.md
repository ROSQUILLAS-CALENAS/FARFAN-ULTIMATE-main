# Canonical Flow Organization

This directory provides a canonical view of the deterministic pipeline organized by stages. Files follow Python module naming conventions with descriptive names rather than numeric prefixes.

## Stage Codes

- `I` = ingestion_preparation
- `X` = context_construction
- `K` = knowledge_extraction
- `A` = analysis_nlp
- `L` = classification_evaluation
- `R` = search_retrieval
- `O` = orchestration_control
- `G` = aggregation_reporting
- `T` = integration_storage
- `S` = synthesis_output

All files have been renamed to follow Python module naming conventions, removing numeric prefixes and using descriptive alphabetic names.

## Index

Files are organized by stage with descriptive names:

### I - Ingestion Preparation
- `pdf_reader.py` (formerly `01I_pdf_reader.py`)
- `advanced_loader.py` (formerly `02I_advanced_loader.py`)
- `feature_extractor.py` (formerly `03I_feature_extractor.py`)
- `normative_validator.py` (formerly `04I_normative_validator.py`)
- `raw_data_generator.py` (formerly `05I_raw_data_generator.py`)

### X - Context Construction  
- `immutable_context.py` (formerly `05X_immutable_context.py`)
- `context_adapter.py` (formerly `06X_context_adapter.py`)
- `lineage_tracker.py` (formerly `07X_lineage_tracker.py`)

### K - Knowledge Extraction
- `advanced_knowledge_graph_builder.py` (formerly `08K_Advanced_Knowledge_Graph_Builder_Component_for_Semantic_Inference_Engine.py`)
- `causal_graph.py` (formerly `09K_causal_graph.py`)
- `causal_dnp_framework.py` (formerly `10K_causal_dnp_framework.py`)
- `embedding_builder.py` (formerly `11K_embedding_builder.py`)
- `embedding_generator.py` (formerly `12K_embedding_generator.py`)

### A - Analysis NLP
- `adaptive_analyzer.py` (formerly `13A_adaptive_analyzer.py`)
- `question_analyzer.py` (formerly `14A_question_analyzer.py`)
- `implementacion_mapeo.py` (formerly `15A_implementacion_mapeo.py`)
- `evidence_processor.py` (formerly `16A_evidence_processor.py`)
- `extractor_evidencias_contextual.py` (formerly `17A_EXTRACTOR_DE_EVIDENCIAS_CONTEXTUAL.py`)
- `evidence_validation_model.py` (formerly `18A_evidence_validation_model.py`)
- `evaluation_driven_processor.py` (formerly `19A_evaluation_driven_processor.py`)
- `dnp_alignment_adapter.py` (formerly `20A_dnp_alignment_adapter.py`)

### L - Classification Evaluation
- `adaptive_scoring_engine.py` (formerly `21L_adaptive_scoring_engine.py`)
- `score_calculator.py` (formerly `22L_score_calculator.py`)
- `conformal_risk_control.py` (formerly `23L_conformal_risk_control.py`)

### O - Orchestration Control
- `deterministic_router.py` (formerly `24O_deterministic_router.py`)
- `evidence_router.py` (formerly `25O_evidence_router.py`)
- `decision_engine.py` (formerly `26O_decision_engine.py`)
- `adaptive_controller.py` (formerly `27O_adaptive_controller.py`)
- `confluent_orchestrator.py` (formerly `37O_confluent_orchestrator.py`)
- `core_orchestrator.py` (formerly `38O_core_orchestrator.py`)
- `enhanced_core_orchestrator.py` (formerly `39O_enhanced_core_orchestrator.py`)
- `distributed_processor.py` (formerly `40O_distributed_processor.py`)
- `airflow_orchestrator.py` (formerly `41O_airflow_orchestrator.py`)
- `circuit_breaker.py` (formerly `42O_circuit_breaker.py`)
- `backpressure_manager.py` (formerly `43O_backpressure_manager.py`)
- `alert_system.py` (formerly `44O_alert_system.py`)
- `exception_monitoring.py` (formerly `45O_exception_monitoring.py`)
- `exception_telemetry.py` (formerly `46O_exception_telemetry.py`)
- `contract_validator.py` (formerly `47O_contract_validator.py`)
- `constraint_validator.py` (formerly `48O_constraint_validator.py`)
- `rubric_validator.py` (formerly `49O_rubric_validator.py`)

### R - Search Retrieval
- `lexical_index_base.py` (formerly `28R_lexical_index.py`)
- `vector_index.py` (formerly `29R_vector_index.py`)
- `hybrid_retriever.py` (formerly `30R_hybrid_retriever.py`)
- `reranker.py` (formerly `31R_reranker.py`)
- `hybrid_retrieval_core.py` (formerly `32R_hybrid_retrieval.py`)
- `deterministic_hybrid_retrieval.py` (formerly `33R_deterministic_hybrid_retrieval.py`)
- `hybrid_retrieval_bridge.py` (formerly `34R_hybrid_retrieval_bridge.py`)
- `lexical_index.py` (formerly `35R_lexical_index.py`)
- `intelligent_recommendation_engine.py` (formerly `36R_intelligent_recommendation_engine.py`)

### S - Synthesis Output
- `answer_synthesizer.py` (formerly `50S_answer_synthesizer.py`)
- `answer_formatter.py` (formerly `51S_answer_formatter.py`)

### G - Aggregation Reporting
- `report_compiler.py` (formerly `52G_report_compiler.py`)
- `meso_aggregator.py` (formerly `53G_meso_aggregator.py`)

### T - Integration Storage
- `metrics_collector.py` (formerly `53T_metrics_collector.py`)
- `analytics_enhancement.py` (formerly `54T_analytics_enhancement.py`)
- `feedback_loop.py` (formerly `55T_feedback_loop.py`)
- `compensation_engine.py` (formerly `56T_compensation_engine.py`)
- `optimization_engine.py` (formerly `57T_optimization_engine.py`)

### Mathematical Enhancers
- `ingestion_enhancer.py` (formerly `math_stage01_ingestion_enhancer.py`)
- `context_enhancer.py` (formerly `math_stage02_context_enhancer.py`)
- `knowledge_enhancer.py` (formerly `math_stage03_knowledge_enhancer.py`)
- `analysis_enhancer.py` (formerly `math_stage04_analysis_enhancer.py`)
- `scoring_enhancer.py` (formerly `math_stage05_scoring_enhancer.py`)
- `retrieval_enhancer.py` (formerly `math_stage06_retrieval_enhancer.py`)
- `orchestration_enhancer.py` (formerly `math_stage07_orchestration_enhancer.py`)
- `aggregation_enhancer.py` (formerly `math_stage11_aggregation_enhancer.py`)
- `integration_enhancer.py` (formerly `math_stage12_integration_enhancer.py`)

<!-- ADV_GRAPHICS_STACK:BEGIN v1 -->
## Advanced Graphics Stack — Innovative Holistic Visuals

This section provides a multi-perspective, advanced visualization of the EGW Query Expansion and Hybrid Retrieval system. The diagrams are designed to be composable, auditable, and implementation-agnostic.

### 1) System Holomap (Architecture Overview)
```mermaid
flowchart LR
  %% Clusters
  subgraph Retrieval["Hybrid Retrieval"]
    A[User Query] --> B{Router ρ}
    B -->|Sparse| S[(SPLADE/BM25)]
    B -->|Dense| D[(E5/FAISS)]
    B -->|Late| L[(ColBERTv2)]
  end
  subgraph OT["EGW Optimal Transport"]
    Q[Query Graph] --- C[Corpus Graph]
    Q -->|ε, λ| T[Transport Plan Π*]
  end
  subgraph Assurance["Contracts & Risk"]
    V1[[Routing Contract ρ]]
    V2[[Conformal Risk]]
    V3[[MCC/BMC]]
  end

  A --> Q
  S & D & L --> C
  T --> R[Expanded Canonical Queries]
  R --> Index[Hybrid Index]
  R --> Answer[Answer Synthesizer]
  V1 -. audits .- B
  V2 -. certify .- Answer
  V3 -. certify .- Pipeline
```

### 2) Deterministic Routing Sequence
```mermaid
sequenceDiagram
  participant U as User
  participant R as Deterministic Router ρ
  participant OT as EGW Engine
  participant H as Hybrid Index
  participant V as Validators (ρ, MCC/BMC, Conformal)
  participant S as Synthesizer
  U->>R: q, Θ, σ, κ
  R->>OT: pattern alignment(q)
  OT-->>R: Π*, barycenter
  R->>H: retrieve(sparse | dense | late)
  H-->>R: candidates
  R->>V: attest(trace, hashes)
  V-->>R: certificate
  R->>S: context, certificate
  S-->>U: answer + lineage
```

### 3) Evidence/Context State Machine
```mermaid
stateDiagram-v2
  [*] --> Ingested
  Ingested --> Normalized
  Normalized --> Indexed: hybrid
  Indexed --> Routed: A* deterministic
  Routed --> Expanded: OT barycenter
  Expanded --> Validated: contracts + conformal
  Validated --> Answered
  Answered --> [*]
```

### 4) Component Metamodel
```mermaid
classDiagram
  class Router {
    +route(q, Θ, σ, κ)
    +deterministicAStar()
    +trace()
  }
  class EGWEngine {
    +align(Q, C)
    +barycenter()
    +stability(ε, λ)
  }
  class HybridIndex {
    +sparse()
    +dense()
    +lateInteraction()
  }
  class Validator {
    +routingContract()
    +conformalRisk()
    +mcc_bmc()
  }
  class Synthesizer {
    +compose()
    +verifyLineage()
  }
  Router --> EGWEngine
  Router --> HybridIndex
  Router --> Validator
  Router --> Synthesizer
```

### 5) Retrieval ER Model
```mermaid
erDiagram
  QUERY ||--o{ CANONICAL_QUERY : expands_to
  CANONICAL_QUERY }o--|| INDEX : indexed_in
  QUERY }o--o{ EVIDENCE : supports
  EVIDENCE ||--o{ CERTIFICATE : yields
  CERTIFICATE ||--|| CONTRACT : attests
  INDEX ||--o{ RETRIEVAL : produces
  RETRIEVAL }o--|| SYNTHESIS : feeds
```

### 6) Pipeline Timeline (Gantt)
```mermaid
gantt
  dateFormat  X
  title Pipeline Phases (Logical Timeline)
  section Preparation
  Ingest/Normalize        :a1, 0, 10
  Build Graphs (Q,C)      :a2, 10, 30
  section Alignment
  EGW Transport Solve     :b1, 30, 50
  Barycenter              :b2, 50, 60
  section Retrieval
  Sparse/Dense/Late       :c1, 60, 80
  section Assurance
  Contracts + Conformal   :d1, 80, 90
  section Synthesis
  Answer + Lineage        :e1, 90, 100
```

> Tip: GitHub renders Mermaid in Markdown by default. If a specific viewer does not support Mermaid, consider using Mermaid live editors to export PNG/SVG equivalents.

<!-- ADV_GRAPHICS_STACK:END v1 -->

<!-- ACADEMIC_ESSAY:BEGIN v1 -->
# Technological Essay — Deterministic EGW Query Expansion and Hybrid Retrieval

## Abstract
The present document advances a comprehensive, academically grounded exposition of a deterministic information retrieval pipeline that integrates Entropic Gromov–Wasserstein (EGW) optimal transport for query expansion with hybrid sparse–dense retrieval. We articulate the theoretical premises of pattern alignment under entropic regularization, formalize determinism via routing and ordering contracts, and explain how conformal risk control, monotone compliance, and evidence lineage produce auditable guarantees from ingestion to synthesis. The pipeline is engineered to be reproducible to the byte, with fixed seeds, stable tie-breaking, and canonical hashing, supporting replay-equivalent snapshots. We discuss design trade-offs, computational complexity, and governance primitives that convert probabilistic components into verifiable, production-grade systems.

## Introduction
Modern retrieval systems frequently rely on heuristic fusion of lexical and embedding-based signals, which may drift, exhibit non-determinism under concurrency, or degrade under domain shift. This project proposes a counterpoint: an end-to-end deterministic orchestration that elevates auditability and scientific reproducibility to first-class system goals. The approach combines EGW-based alignment to map queries to corpus structure with carefully specified contracts that regulate routing, ordering, idempotency, and risk calibration. Rather than treating determinism as an afterthought, we encode it as an invariant backed by tests, certificates, and Merkle-chained traces so that identical inputs and hyper-parameters provably yield identical outputs.

## Theoretical Foundations
Our foundations draw from optimal transport, information theory, and graph alignment. Entropic Gromov–Wasserstein provides a geometry-aware mechanism to align a query graph and a corpus graph while controlling stability through entropy (ε) and coupling strength (λ). The induced barycenter produces canonical expansions that preserve relational structure rather than isolated token similarity. On top of this, we layer deterministic routing modeled as an A* search whose cost function and tie-breaking are fully specified, ensuring byte-level reproducibility. The theory of conformal prediction supplies distribution-free coverage guarantees; monotone consistency and budget monotonicity establish that support cannot degrade under additional non-contradictory evidence or enlarged feasible budgets. Together, these elements constitute a principled scaffold for reliable retrieval under changing conditions.

## System Architecture
The system decomposes into modular components connected through typed interfaces and verifiable contracts: a context normalizer constructs immutable snapshots; an alignment engine solves EGW to extract transport plans and barycenters; a hybrid index performs lexical, dense, and late-interaction retrieval; validators apply routing, ordering, idempotency, stability, and risk contracts; and a synthesizer composes answers with explicit lineage. Deterministic seeds are derived from trace identifiers, while all sorting operations employ stable, lexicographic tie-breakers on content hashes and module identifiers. Each module emits structured telemetry and cryptographic digests, permitting replay, regression detection, and drift analysis without reliance on hidden state or wall-clock nondeterminism.

## Methodology
We operationalize determinism through explicit algorithmic choices and serialized evidence. Routing employs a deterministic A* variant with invariant heuristics and lexicographic tie-breaking κ to resolve exact-score ties. EGW alignment is configured with fixed ε, λ, iteration budgets, and convergence tolerances; the full transport plan Π and diagnostics are serialized for audit. Hybrid retrieval uses reproducible indices and de-duplication by content hash, while ordering adheres to a total ordering contract that guarantees stable output rankings. Conformal risk calibration fixes α, partitioning schemes, and seeds, resulting in certificates that can be attached to synthesized answers. The methodology treats every intermediate product—queries, candidates, transport plans, rankings, certificates—as evidence with verifiable provenance.

## Evaluation and Metrics
Evaluation prioritizes determinism, calibration, and structural fidelity in addition to standard retrieval quality. We measure exact replay equality on snapshots, hash-level identity of routes and rankings, and certificate validity rates under controlled perturbations. Alignment quality is assessed via transport stability and barycentric consistency, whereas retrieval performance is profiled across sparse, dense, and late-interaction backends with ablations isolating each contract’s contribution. We further monitor concurrency determinism by verifying that parallelizable subroutines employ associative, commutative reducers or explicit pre-sorting, avoiding nondeterministic reductions. These metrics collectively quantify not only how well the system retrieves but how reliably it can be reproduced and audited.

## Reproducibility and Governance
Reproducibility is enforced through snapshot immutability, dependency audits, byte-compilation checks, and structured project analysis. A Merkle-chained trace logger records the inputs, parameters Θ, context σ, and output digests at each stage, enabling replay audits and forensic debugging. Governance documents and certificates—covering routing, alignment stability, calibration coverage, and refusal conditions—are bundled with outputs to form a verifiable dossier. This governance layer empowers operators to reason about system behavior in adversarial or shifting environments, aligning engineering practice with the scientific norm of falsifiability and repeatability.

## Limitations and Threats to Validity
While determinism bolsters trust and auditability, it can constrain throughput when naive serialization is imposed; therefore, we exploit safe parallelism only where algebraic properties guarantee identical results. Entropic regularization introduces bias that trades variance for stability; tuning ε and λ requires sensitivity analyses to avoid oversmoothing semantic distinctions. Calibration guarantees depend on exchangeability assumptions that can be stressed under covariate shift; we mitigate this by monitoring shift diagnostics and enforcing fail-closed refusal contracts when preconditions are violated. Finally, reproducibility metadata must be maintained with care to avoid accidental divergence between documentation and runtime behavior.

## Related Work
This work synthesizes threads from optimal transport in machine learning, structure-preserving retrieval, deterministic systems design, and distribution-free uncertainty quantification. Prior art on hybrid retrieval and rank fusion often emphasizes empirical gains without specifying deterministic contracts, leaving gaps in auditability. Our contribution reframes these components as contract-governed modules and integrates conformal calibration and monotone compliance to furnish guarantees that are actionable in production contexts rather than solely in laboratory evaluations.

## Ethics and Safety
Retrieval and synthesis systems can amplify biases, leak sensitive information, or hallucinate unsupported content. Our pipeline’s evidence-centric architecture enforces lineage and idempotency, making it harder for spurious or unverifiable content to traverse gates. Conformal certificates articulate uncertainty transparently, and refusal contracts ensure that missing prerequisites result in typed, minimal disclosures rather than speculative outputs. Auditable traces facilitate redress mechanisms and enable compliance with regulatory standards concerning data provenance and reproducibility.

## Conclusion and Future Work
By treating determinism, auditability, and calibration as core design constraints, we demonstrate that modern retrieval can be both high-performing and scientifically rigorous. Future work includes adaptive EGW schemes with provable stability under bounded distribution shifts, broader benchmarking across multilingual corpora, and formal verification of routing and ordering implementations. We also intend to expand tool support for continuous certification so that every production run maintains an automatically generated dossier of evidence, metrics, and risk guarantees.

## References
- Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal transport. NIPS.
- Peyré, G., & Cuturi, M. (2019). Computational Optimal Transport. Foundations and Trends in ML.
- Bruch, S., Han, S., Bendersky, M., et al. (2023). A principled framework for optimal rank fusion. WWW.
- Shafer, G., & Vovk, V. (2008). A tutorial on conformal prediction. JMLR.
- Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World. Springer.

## Glossary of Symbols
Θ (theta): hyper-parameters; σ (sigma): context digest; κ (kappa): lexicographic tie-breaker; ε (epsilon): entropic regularizer; λ (lambda): coupling strength; Π (pi): transport plan; ρ (rho): routing function. These symbols appear throughout the pipeline specification and are serialized in traces and certificates to support byte-identical replay and audit.

<!-- ACADEMIC_ESSAY:END v1 -->

# Human Rights Assessment Methodology

## Dimensional Framework

### Four-Cluster Processing System (C1-C4)

The methodology implements a deterministic four-cluster processing system that ensures comprehensive coverage:

#### Cluster Structure

```
C1: Individual Rights and Freedoms
├── Life and Security (P1)
├── Human Dignity (P2)
└── Equality and Non-discrimination (P3)

C2: Social and Economic Rights
├── Basic Services (P5)
├── Inclusive Economic Development (P7)
└── Environmental Protection (P6)

C3: Political and Cultural Rights
├── Citizen Participation (P4)
├── Cultural and Territorial Rights (P8)
└── Access to Justice (P9)

C4: Governance and Accountability
└── Transparency and Accountability (P10)
```

#### Processing Order

The system processes clusters in fixed sequential order: C1 → C2 → C3 → C4

Each cluster undergoes:
1. Questionnaire application
2. Evidence extraction and linking
3. Micro-level response generation
4. Cross-cluster validation
5. Scoring and assessment

## Scoring Mechanisms

### Multi-Level Scoring Framework

#### 1. Evidence-Level Scoring

```python
evidence_score = {
    "relevance_score": float,  # 0.0 - 1.0
    "quality_score": float,    # 0.0 - 1.0 
    "completeness_score": float, # 0.0 - 1.0
    "reliability_score": float   # 0.0 - 1.0
}
```

#### 2. Question-Level Assessment

For each question (P1-P10), the system calculates:

- **Evidence Count**: Number of relevant evidence pieces
- **Average Relevance**: Mean relevance score across evidence
- **Compliance Indicator**: 
  - `CUMPLE` (≥2 high-quality evidence pieces)
  - `CUMPLE_PARCIAL` (1 evidence piece)
  - `NO_CUMPLE` (0 evidence pieces)

#### 3. Cluster-Level Aggregation

```python
cluster_score = {
    "total_questions": 10,
    "compliant_questions": int,
    "partially_compliant": int,
    "non_compliant": int,
    "overall_compliance_rate": float,
    "weighted_score": float
}
```

#### 4. Global Human Rights Score

```python
global_score = sum(cluster_weighted_scores) / 4
```

With DNP validation corrections applied:
- **HIGH** severity: global_score × 0.8
- **CRITICAL** severity: global_score × 0.6

### Adaptive Scoring Engine

The system includes an adaptive scoring component that:

1. **Contextual Adjustment**: Adjusts scores based on municipal context
2. **Historical Learning**: Uses past assessment data for prediction
3. **Risk Assessment**: Identifies critical compliance gaps
4. **Causal Analysis**: Evaluates intervention-outcome relationships

#### DNP Baseline Standards

```python
dnp_baselines = {
    "minimum_compliance_threshold": 0.6,
    "critical_gap_threshold": 0.3,
    "evidence_quality_minimum": 0.5,
    "participation_requirement": 0.7
}
```

## Technical Implementation

### Evidence Processing Pipeline

#### 1. PDF Content Extraction

```python
class PDFProcessor:
    def extract_content(self, pdf_path):
        # OCR and text extraction
        # Structure parsing
        # Metadata generation
        return structured_content
```

#### 2. Evidence Linking

The system maps PDF content to human rights indicators through:

```python
def link_evidence_to_questions(content, question_id):
    """
    Maps extracted PDF content to specific human rights questions
    
    Args:
        content: Extracted PDF text/structure
        question_id: Target question (P1-P10)
    
    Returns:
        Evidence links with relevance scores
    """
    relevance_score = calculate_semantic_similarity(content, question_id)
    quality_metrics = assess_evidence_quality(content)
    
    return {
        "evidence_id": generate_evidence_id(content, question_id),
        "relevance_score": relevance_score,
        "quality_metrics": quality_metrics,
        "content_summary": extract_summary(content)
    }
```

#### 3. Structured Evidence Generation

```python
@dataclass
class StructuredEvidence:
    evidence_id: str
    question_id: str
    cluster_id: str
    content_text: str
    relevance_score: float
    quality_metrics: Dict[str, float]
    source_metadata: SourceMetadata
    validation_result: ValidationResult
    traceability_path: List[str]
```

### Integration with Existing Components

#### 1. Evidence System Integration

The human rights assessment integrates with the existing evidence processing system:

```python
# In evidence_processor.py
class EvidenceProcessor:
    def process_human_rights_evidence(self, chunks, metadata_list, question_id):
        """Process evidence chunks for human rights assessment"""
        structured_evidence = []
        
        for chunk, metadata in zip(chunks, metadata_list):
            evidence = self.create_structured_evidence(
                chunk=chunk,
                metadata=metadata,
                question_id=question_id,
                evidence_type=EvidenceType.HUMAN_RIGHTS_INDICATOR
            )
            structured_evidence.append(evidence)
            
        return structured_evidence
```

#### 2. Retrieval Engine Integration

Human rights queries are processed through the hybrid retrieval system:

```python
# Integration with retrieval_engine.py
def process_human_rights_query(query, question_id, cluster_id):
    """
    Process human rights queries through hybrid retrieval
    
    Returns evidence ranked by relevance to specific human rights dimensions
    """
    results = hybrid_retrieval_engine.search(
        query=query,
        filters={"cluster_id": cluster_id, "question_id": question_id},
        rerank_for_human_rights=True
    )
    
    return results
```

## PDF Content Mapping

### Extraction to Assessment Pipeline

#### Phase 1: Content Extraction
```
PDF Document → OCR/Text Extraction → Structure Parsing → Content Segmentation
```

#### Phase 2: Semantic Analysis
```
Content Segments → NLP Processing → Concept Extraction → Rights Mapping
```

#### Phase 3: Evidence Generation
```
Rights Mapping → Evidence Linking → Quality Assessment → Structured Output
```

### Example Transformation

**Input PDF Text:**
```
"El municipio implementará programas de seguridad ciudadana mediante 
patrullajes comunitarios y sistemas de videovigilancia para garantizar 
la protección de los habitantes en zonas de alto riesgo."
```

**Processed Evidence:**
```python
{
    "evidence_id": "ev_C1_P1_a1b2c3d4",
    "question_id": "P1",
    "cluster_id": "C1", 
    "content_text": "El municipio implementará programas de seguridad...",
    "relevance_score": 0.89,
    "quality_metrics": {
        "completeness": 0.75,
        "specificity": 0.82,
        "actionability": 0.91
    },
    "compliance_indicator": "CUMPLE",
    "mapped_concepts": ["seguridad_ciudadana", "proteccion_habitantes", "patrullajes"],
    "human_rights_dimensions": ["derecho_vida", "derecho_seguridad"]
}
```

**Quantified Score:**
- **Question P1 Score**: 0.89 (based on evidence relevance and quality)
- **Cluster C1 Contribution**: 0.297 (0.89 × 0.33 weight)
- **Global Impact**: +0.074 (0.297 × 0.25 cluster weight)

## Evidence Processing Pipeline

### Multi-Stage Processing

#### Stage 1: Ingestion
```python
def ingest_pdf_content(pdf_path, context):
    """
    Initial PDF processing and content extraction
    """
    content = extract_pdf_text(pdf_path)
    structured_data = parse_document_structure(content)
    metadata = generate_source_metadata(pdf_path, structured_data)
    
    return {
        "content": structured_data,
        "metadata": metadata,
        "processing_timestamp": datetime.now()
    }
```

#### Stage 2: Evidence Detection
```python
def detect_human_rights_evidence(content, questionnaire):
    """
    Identify potential human rights evidence in content
    """
    evidence_candidates = []
    
    for question_id, question_text in questionnaire.items():
        matches = semantic_search(content, question_text)
        
        for match in matches:
            if match.relevance_score > 0.5:
                evidence_candidates.append({
                    "question_id": question_id,
                    "text_segment": match.text,
                    "relevance_score": match.relevance_score,
                    "position": match.document_position
                })
    
    return evidence_candidates
```

#### Stage 3: Quality Assessment
```python
def assess_evidence_quality(evidence_candidate):
    """
    Multi-dimensional evidence quality assessment
    """
    quality_metrics = {
        "completeness": assess_completeness(evidence_candidate),
        "specificity": assess_specificity(evidence_candidate), 
        "verifiability": assess_verifiability(evidence_candidate),
        "temporal_relevance": assess_temporal_relevance(evidence_candidate),
        "institutional_source": assess_source_reliability(evidence_candidate)
    }
    
    overall_quality = weighted_average(quality_metrics, quality_weights)
    
    return {
        "individual_metrics": quality_metrics,
        "overall_quality": overall_quality,
        "quality_tier": classify_quality_tier(overall_quality)
    }
```

## Compliance Assessment

### Multi-Level Assessment Framework

#### 1. Question-Level Compliance
```python
def assess_question_compliance(question_id, evidence_list):
    """
    Assess compliance for individual human rights questions
    """
    if not evidence_list:
        return {
            "compliance_level": "NO_CUMPLE",
            "score": 0.0,
            "evidence_count": 0,
            "quality_summary": "No evidence found"
        }
    
    high_quality_evidence = [e for e in evidence_list if e.quality_score >= 0.7]
    medium_quality_evidence = [e for e in evidence_list if 0.5 <= e.quality_score < 0.7]
    
    if len(high_quality_evidence) >= 2:
        compliance_level = "CUMPLE"
        score = min(0.9, np.mean([e.relevance_score for e in high_quality_evidence]))
    elif len(high_quality_evidence) >= 1 or len(medium_quality_evidence) >= 2:
        compliance_level = "CUMPLE_PARCIAL" 
        score = 0.6 * np.mean([e.relevance_score for e in evidence_list])
    else:
        compliance_level = "NO_CUMPLE"
        score = 0.3 * np.mean([e.relevance_score for e in evidence_list])
    
    return {
        "compliance_level": compliance_level,
        "score": score,
        "evidence_count": len(evidence_list),
        "high_quality_count": len(high_quality_evidence)
    }
```

#### 2. Cluster-Level Aggregation
```python
def aggregate_cluster_scores(cluster_questions):
    """
    Aggregate question scores within a cluster
    """
    question_scores = [q["score"] for q in cluster_questions.values()]
    compliance_levels = [q["compliance_level"] for q in cluster_questions.values()]
    
    cluster_score = {
        "average_score": np.mean(question_scores),
        "weighted_score": calculate_weighted_cluster_score(cluster_questions),
        "compliance_distribution": {
            "CUMPLE": compliance_levels.count("CUMPLE"),
            "CUMPLE_PARCIAL": compliance_levels.count("CUMPLE_PARCIAL"), 
            "NO_CUMPLE": compliance_levels.count("NO_CUMPLE")
        },
        "overall_compliance_rate": compliance_levels.count("CUMPLE") / len(compliance_levels)
    }
    
    return cluster_score
```

#### 3. Global Assessment with DNP Corrections
```python
def calculate_global_human_rights_score(cluster_scores, dnp_validation):
    """
    Calculate final global human rights compliance score
    """
    base_score = np.mean([cluster["weighted_score"] for cluster in cluster_scores.values()])
    
    # Apply DNP validation corrections
    if dnp_validation.get("severity_assessment") == "HIGH":
        corrected_score = base_score * 0.8
    elif dnp_validation.get("severity_assessment") == "CRITICAL":
        corrected_score = base_score * 0.6
    else:
        corrected_score = base_score
    
    # Classification based on corrected score
    if corrected_score >= 0.8:
        classification = "EXCELLENT_COMPLIANCE"
    elif corrected_score >= 0.6:
        classification = "ADEQUATE_COMPLIANCE"
    elif corrected_score >= 0.4:
        classification = "NEEDS_IMPROVEMENT"
    else:
        classification = "CRITICAL_NON_COMPLIANCE"
    
    return {
        "base_score": base_score,
        "corrected_score": corrected_score,
        "classification": classification,
        "dnp_severity": dnp_validation.get("severity_assessment", "UNKNOWN")
    }
```

## Reproducibility and Auditability

### Audit Trail Generation

#### 1. Processing Trace
```python
def generate_processing_trace(assessment_run):
    """
    Generate comprehensive audit trail for assessment
    """
    return {
        "assessment_id": assessment_run.id,
        "timestamp": assessment_run.timestamp,
        "input_documents": [doc.path for doc in assessment_run.documents],
        "processing_stages": [
            {
                "stage": "pdf_extraction",
                "duration": assessment_run.extraction_duration,
                "documents_processed": len(assessment_run.documents),
                "errors": assessment_run.extraction_errors
            },
            {
                "stage": "evidence_detection", 
                "evidence_candidates_found": assessment_run.evidence_candidates_count,
                "questions_with_evidence": assessment_run.questions_with_evidence,
                "quality_filters_applied": assessment_run.quality_filters
            },
            {
                "stage": "cluster_processing",
                "clusters_processed": ["C1", "C2", "C3", "C4"],
                "questionnaire_applications": 4,
                "cross_cluster_validations": assessment_run.cross_validations
            },
            {
                "stage": "scoring_and_assessment",
                "base_scores": assessment_run.base_scores,
                "dnp_corrections_applied": assessment_run.dnp_corrections,
                "final_scores": assessment_run.final_scores
            }
        ]
    }
```

#### 2. Evidence Traceability
```python
def generate_evidence_traceability(evidence_item):
    """
    Generate full traceability path for evidence items
    """
    return {
        "evidence_id": evidence_item.evidence_id,
        "source_document": evidence_item.source_metadata.document_id,
        "extraction_method": evidence_item.extraction_method,
        "processing_pipeline": evidence_item.processing_pipeline,
        "quality_assessments": evidence_item.quality_history,
        "human_validation": evidence_item.human_validation_status,
        "linked_questions": evidence_item.linked_questions,
        "cluster_assignments": evidence_item.cluster_assignments,
        "score_contributions": evidence_item.score_contributions
    }
```

#### 3. Validation Checkpoints
```python
def validate_assessment_integrity(assessment_results):
    """
    Validate assessment integrity and completeness
    """
    validations = {
        "cluster_completeness": validate_all_clusters_processed(assessment_results),
        "question_coverage": validate_all_questions_addressed(assessment_results),
        "evidence_quality": validate_evidence_quality_standards(assessment_results),
        "score_consistency": validate_score_calculation_consistency(assessment_results),
        "dnp_compliance": validate_dnp_standard_adherence(assessment_results)
    }
    
    overall_validity = all(validations.values())
    
    return {
        "is_valid": overall_validity,
        "validation_details": validations,
        "integrity_score": sum(validations.values()) / len(validations)
    }
```

### Configuration Management

The system maintains strict configuration versioning for reproducible assessments:

```python
assessment_config = {
    "questionnaire_version": "decalogo_v1.0",
    "scoring_algorithm_version": "adaptive_v2.1", 
    "dnp_standards_version": "2025.1",
    "evidence_quality_thresholds": {
        "minimum_relevance": 0.5,
        "minimum_quality": 0.6,
        "high_quality_threshold": 0.7
    },
    "cluster_weights": {
        "C1": 0.25,  # Individual Rights
        "C2": 0.30,  # Social/Economic Rights  
        "C3": 0.25,  # Political/Cultural Rights
        "C4": 0.20   # Governance
    },
    "dnp_correction_factors": {
        "HIGH": 0.8,
        "CRITICAL": 0.6
    }
}
```

This comprehensive methodology ensures consistent, auditable, and reproducible human rights assessments while maintaining integration with the existing PDF analysis pipeline and evidence processing components.


## Canonical Source Transcription

Provenance note: The following sections are a verbatim transcription of in-repository attached documents to ensure integrity and transparency. Source files and retrieval datetime:
- Source 1: Cuestionario Original de la Metodología.md (retrieved 2025-08-26 20:18 local time)
- Source 2: Decálogo de Derechos Humanos_ Puntos y Clústeres.md (retrieved 2025-08-26 20:18 local time)

No rewording or editorial changes have been applied below; formatting preserved as-is.

---

### Cuestionario Original de la Metodología (verbatim)

# Cuestionario Original de la Metodología

A continuación, se presentan las preguntas de verificación y criterios de cada dimensión de evaluación (DE-1 a DE-4) tal como se describen en la metodología original, sin cruce con el Decálogo de Derechos Humanos.

## DE-1: Lógica de Intervención y Coherencia Interna

**Cuestionario de verificación:** Evalúa si el PDT presenta una conexión lógica y coherente entre los problemas identificados en el diagnóstico, las causas y consecuencias planteadas, las actividades y productos propuestos, y los resultados esperados, siguiendo la cadena de valor del DNP. Escala de respuesta: Sí / Parcial / No / NI (No identificado tras búsqueda exhaustiva).

*   **Q1:** ¿El PDT define productos medibles alineados con la prioridad?
    *   **Evidencia esperada:** Producto descrito, valor meta, año de cumplimiento.
*   **Q2:** ¿Las metas de producto incluyen responsable institucional?
    *   **Evidencia esperada:** Cargo o dependencia explícitamente nombrada.
*   **Q3:** ¿Formula resultados medibles con línea base y meta al 2027?
    *   **Evidencia esperada:** Indicador de resultado con línea base y meta cuantitativa.
*   **Q4:** ¿Resultados y productos están lógicamente vinculados según la cadena de valor?
    *   **Evidencia esperada:** Texto o cuadro que muestre cómo los productos llevan a los resultados esperados.
*   **Q5:** ¿El impacto esperado está definido y alineado al Decálogo?
    *   **Evidencia esperada:** Indicador de bienestar proyectado con meta clara.
*   **Q6:** ¿Existe una explicación explícita de la lógica de intervención completa?
    *   **Evidencia esperada:** Párrafo o diagrama que describa la racionalidad de la secuencia productos→resultados→impactos esperados.

## DE-2: Inclusión Temática (IT)

**Cuestionario de verificación:** Se compone de cinco subdimensiones temáticas. Cada una incluye criterios verificables con escala Sí / No, que se responden exclusivamente en función de evidencia concreta y localizada dentro del PDT.

### Subdimensión 2.1: Diagnóstico con enfoque de derechos (25%)

*   **D1:** Línea base 2023 con fuente citada
    *   **Evidencia esperada:** Tabla o gráfico con valor de 2023 + referencia (DANE, SISBEN, etc.)
*   **D2:** Serie histórica ≥ 5 años
    *   **Evidencia esperada:** Muestra de al menos cinco puntos temporales (2018-2023)
*   **D3:** Identificación de causas directas
    *   **Evidencia esperada:** Texto analítico que explique por qué ocurre la brecha
*   **D4:** Identificación de causas estructurales
    *   **Evidencia esperada:** Mención explícita de determinantes (pobreza, conflicto, género, etc.)
*   **D5:** Brechas territoriales detalladas
    *   **Evidencia esperada:** Comparación intra-municipal (urbano/rural, comunas, corregimientos)
*   **D6:** Grupos poblacionales afectados identificados
    *   **Evidencia esperada:** Cuadro o lista que mencione niñez, mujeres, pueblos indígenas, etc.

### Subdimensión 2.2: Alineación Estratégica con Marcos de Largo Plazo y Ambición (30%)

*   **O1:** Objetivo específico alineado con transformaciones del PND
    *   **Evidencia esperada:** Referencia explícita a catalizadores o transformaciones del PND
*   **O2:** Indicador de resultado con línea base y meta transformadora
    *   **Evidencia esperada:** Valor inicial + meta cuantificada a 2027 que busque cambio sistémico
*   **O3:** Meta que aborde problemas estructurales del territorio
    *   **Evidencia esperada:** Indicador que enfrente causas profundas identificadas en diagnóstico
*   **O4:** Relación explícita con múltiples ODS
    *   **Evidencia esperada:** Al menos 3 ODS citados y justificados en la ficha del proyecto
*   **O5:** Acción o programa con visión de largo plazo
    *   **Evidencia esperada:** Programa que trascienda el período de gobierno con sostenibilidad
*   **O6:** Articulación con determinantes ambientales del ordenamiento
    *   **Evidencia esperada:** Mención de protección del suelo rural, estructura ecológica, etc.

### Subdimensión 2.3: Territorialización y PPI (25%)

*   **T1:** Proyecto codificado en BPIN o código interno
    *   **Evidencia esperada:** Número BPIN o código visible en el anexo del PDT
*   **T2:** Monto plurianual 2024-2027
    *   **Evidencia esperada:** Valores anuales o acumulado desagregado 2024-2027 especificado
*   **T3:** Fuente de financiación identificada
    *   **Evidencia esperada:** SG-P, Regalías, Propios, Convenios, etc.
*   **T4:** Asignación ≥ 1% del total presupuestal
    *   **Evidencia esperada:** Cálculo explícito dentro del Plan Plurianual de Inversiones (PPI)
*   **T5:** Distribución geográfica o sectorial clara
    *   **Evidencia esperada:** Tabla que desagrega por comuna, corregimiento o sector

### Subdimensión 2.4: Seguimiento y articulación (20%)

*   **S1:** Indicadores de producto y resultado en SUIFP-T
    *   **Evidencia esperada:** Listado con códigos específicos
*   **S2:** Periodicidad de reporte especificada
    *   **Evidencia esperada:** Frecuencia de reporte: mensual, trimestral, anual
*   **S3:** Dependencia responsable de seguimiento
    *   **Evidencia esperada:** Secretaría, oficina u organismo designado explícitamente
*   **S4:** Coordinación con nivel nacional o departamental
    *   **Evidencia esperada:** Convenio, protocolo o comité de articulación

## DE-3: Planificación y Adecuación Presupuestal

**Cuestionario de verificación:** Se plantean ocho preguntas, cada una vinculada a un componente clave de la planificación presupuestal y organizacional. Las respuestas siguen la escala: Sí / Parcial / No / NI.

*   **G1:** ¿Existe identificación de fuentes de financiación diversificadas?
    *   **Evidencia esperada:** Listado de fuentes: recursos propios, SGR, PGN, banca, cooperación, Obras por Impuestos.
*   **G2:** ¿Se presenta distribución presupuestal anualizada?
    *   **Evidencia esperada:** Tabla con montos por año 2024-2027.
*   **A1:** ¿Los montos son coherentes con la ambición de las metas?
    *   **Evidencia esperada:** Análisis de suficiencia presupuestal o justificación de montos.
*   **A2:** ¿Hay estrategia de gestión de recursos adicionales?
    *   **Evidencia esperada:** Plan de consecución de recursos o alianzas documentado.
*   **R1:** ¿Los recursos están trazados en el PPI con códigos?
    *   **Evidencia esperada:** Monto específico + código BPIN o identificador presupuestal.
*   **R2:** ¿Se identifica necesidad de fortalecer capacidades?
    *   **Evidencia esperada:** Mención de requerimientos de personal técnico o institucional.
*   **S1:** ¿El presupuesto está alineado con el plan indicativo?
    *   **Evidencia esperada:** Correspondencia entre PPI y metas del plan indicativo.
*   **S2:** ¿Existe plan de contingencia presupuestal?
    *   **Evidencia esperada:** Escenarios alternativos o fuentes de respaldo identificadas.

## DE-4: Cadena de valor

**Cuestionario de verificación:** Se revisan ocho eslabones clave del ciclo de planeación y ejecución. La respuesta es dicotómica: Sí / No. No se utiliza la categoría "Parcial" para esta dimensión, debido a la naturaleza acumulativa de los elementos. Los ocho eslabones son:

1.  Diagnóstico con línea base y brechas claras.
2.  Causalidad explícita entre productos, resultados e impacto.
3.  Metas formuladas con claridad y ambición transformadora.
4.  Programas o acciones detalladas con responsable y presupuesto.
5.  Territorialización de las intervenciones (geográfica o sectorial).
6.  Vinculación institucional (articulación con sectores o niveles).
7.  Seguimiento con indicadores y calendario definido.
8.  Proyección de impacto o beneficio con alineación al Decálogo.

Cada eslabón se marca "Sí" únicamente si se evidencia de forma inequívoca, con cita localizada. En caso contrario, se marca "No".

---

### Decálogo de Derechos Humanos: Puntos y Clústeres (verbatim)

# Decálogo de Derechos Humanos: Puntos y Clústeres

Based on the advanced prompts catalog, here is the ordered list of all Decálogo points and their corresponding clusters:

## CLUSTER 1: PAZ, SEGURIDAD Y PROTECCIÓN DE DEFENSORES

**Logic of Grouping:** This cluster focuses on human security, protection of life, and territorial peace building. It addresses the dynamics of armed conflict, its victims, and those who defend rights in risk contexts.

### Points included in Cluster 1:
- **Point 1:** Prevención de la violencia y protección de la población frente al conflicto armado y la violencia generada por GDO (Prevention of violence and protection of the population against armed conflict and violence generated by organized armed groups)
- **Point 5:** Derechos de las víctimas y construcción de paz (Rights of victims and peace building)
- **Point 8:** Líderes y defensores de derechos humanos sociales y ambientales (Leaders and defenders of social and environmental human rights)

## CLUSTER 2: DERECHOS SOCIALES FUNDAMENTALES

**Logic of Grouping:** This cluster encompasses basic social rights that guarantee human dignity and quality of life.

### Points included in Cluster 2:
- **Point 2:** Derecho a la salud (Right to health)
- **Point 3:** Derecho a la educación (Right to education)
- **Point 4:** Derecho a la alimentación (Right to food)

## CLUSTER 3: IGUALDAD Y NO DISCRIMINACIÓN

**Logic of Grouping:** This cluster focuses on equality, non-discrimination, and the protection of vulnerable groups.

### Points included in Cluster 3:
- **Point 6:** Derechos de las mujeres (Women's rights)
- **Point 7:** Derechos de niñas, niños y adolescentes (Rights of girls, boys and adolescents)

## CLUSTER 4: DERECHOS TERRITORIALES Y AMBIENTALES

**Logic of Grouping:** This cluster addresses territorial rights, environmental protection, and sustainable development.

### Points included in Cluster 4:
- **Point 9:** Derechos de los pueblos étnicos (Rights of ethnic peoples)
- **Point 10:** Derecho a un ambiente sano (Right to a healthy environment)

## CLUSTER 5: DERECHO A LA VIDA, SEGURIDAD Y CONVIVENCIA

**Logic of Grouping:** This cluster focuses on citizen security, crime prevention, and peaceful coexistence.

### Points included in Cluster 5:
- **Point 1:** Derecho a la vida, seguridad y convivencia (Right to life, security and coexistence)

**Note:** Point 1 appears in both Cluster 1 and Cluster 5 with different focuses - in Cluster 1 it emphasizes prevention of violence related to armed conflict and organized armed groups, while in Cluster 5 it focuses on citizen security and peaceful coexistence.

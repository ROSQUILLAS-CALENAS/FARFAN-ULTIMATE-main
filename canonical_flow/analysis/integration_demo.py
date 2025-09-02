#!/usr/bin/env python3
"""
Integration demo for comprehensive artifact generation system
Demonstrates how to integrate with existing scoring systems and analysis components.
"""

import sys
import os
from pathlib import Path
import json
from typing import Dict, List, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from canonical_flow.analysis.artifact_generator import (
    ArtifactGenerator, QuestionEvaluation, DimensionSummary, PointSummary,
    MesoClusterAnalysis, MacroAlignment, EvidenceReference
)


def create_realistic_decalogo_data(document_stem: str) -> tuple:
    """Create realistic Decálogo evaluation data."""
    
    # Create evidence references for human rights points
    evidence_refs_health = [
        EvidenceReference("EV-HEALTH-001", "institutional_document", "p. 45-47", 
                         "Sistema de salud presenta limitaciones en cobertura rural", 0.88),
        EvidenceReference("EV-HEALTH-002", "field_interview", "p. 12", 
                         "Testimonios de comunidades sobre acceso a servicios", 0.92),
        EvidenceReference("EV-HEALTH-003", "statistical_report", "p. 23", 
                         "Indicadores de mortalidad infantil por región", 0.85)
    ]
    
    evidence_refs_education = [
        EvidenceReference("EV-EDU-001", "census_data", "p. 67", 
                         "Tasas de deserción escolar en zonas de conflicto", 0.94),
        EvidenceReference("EV-EDU-002", "ministry_report", "p. 89", 
                         "Infraestructura educativa insuficiente", 0.87)
    ]
    
    # Create questions for Point 2 (Right to Health)
    health_questions = [
        # DE-1 Productos questions
        QuestionEvaluation("DE-1-Q1", "¿Se cuenta con diagnósticos actualizados sobre el estado del derecho a la salud?", 
                          "Parcial", 0.5, 0.7, 0.8, 0.65, evidence_refs_health),
        QuestionEvaluation("DE-1-Q2", "¿Existen planes de acción específicos para garantizar el derecho a la salud?", 
                          "Sí", 1.0, 0.9, 0.85, 1.08, evidence_refs_health[:2]),
        QuestionEvaluation("DE-1-Q3", "¿Los productos normativos incorporan enfoque diferencial?", 
                          "No", 0.0, 0.4, 0.6, 0.0, evidence_refs_health[1:]),
        
        # DE-2 Diagnóstico questions  
        QuestionEvaluation("DE-2-Q1", "¿Se identifican barreras específicas de acceso a servicios de salud?", 
                          "Sí", 1.0, 0.85, 0.9, 1.05, evidence_refs_health),
        QuestionEvaluation("DE-2-Q2", "¿El diagnóstico incluye análisis de determinantes sociales de la salud?", 
                          "Parcial", 0.5, 0.6, 0.7, 0.58, evidence_refs_health[:1]),
        
        # DE-3 Seguimiento questions
        QuestionEvaluation("DE-3-Q1", "¿Existe sistema de monitoreo de indicadores de salud?", 
                          "Sí", 1.0, 0.9, 0.95, 1.14, evidence_refs_health[2:]),
        QuestionEvaluation("DE-3-Q2", "¿Se realizan evaluaciones periódicas de políticas de salud?", 
                          "Parcial", 0.5, 0.5, 0.8, 0.58, evidence_refs_health[:2]),
        
        # DE-4 Evaluación questions
        QuestionEvaluation("DE-4-Q1", "¿Se evalúa el impacto de las intervenciones en salud?", 
                          "No", 0.0, 0.3, 0.5, 0.0, evidence_refs_health[1:]),
        QuestionEvaluation("DE-4-Q2", "¿Las evaluaciones informan ajustes en políticas públicas?", 
                          "Parcial", 0.5, 0.4, 0.6, 0.45, evidence_refs_health)
    ]
    
    # Create questions for Point 3 (Right to Education)
    education_questions = [
        QuestionEvaluation("DE-1-Q1", "¿Existe diagnóstico sobre acceso y calidad educativa?", 
                          "Sí", 1.0, 0.95, 0.9, 1.11, evidence_refs_education),
        QuestionEvaluation("DE-1-Q2", "¿Hay programas específicos para población vulnerable?", 
                          "Parcial", 0.5, 0.8, 0.7, 0.63, evidence_refs_education),
        
        QuestionEvaluation("DE-2-Q1", "¿Se identifican factores de deserción escolar?", 
                          "Sí", 1.0, 0.9, 0.85, 1.05, evidence_refs_education),
        QuestionEvaluation("DE-2-Q2", "¿El diagnóstico incluye análisis territorial?", 
                          "Parcial", 0.5, 0.7, 0.75, 0.61, evidence_refs_education[:1]),
        
        QuestionEvaluation("DE-3-Q1", "¿Existe seguimiento a indicadores educativos?", 
                          "Sí", 1.0, 0.85, 0.9, 1.05, evidence_refs_education),
        QuestionEvaluation("DE-3-Q2", "¿Se monitorea la infraestructura educativa?", 
                          "Parcial", 0.5, 0.6, 0.65, 0.56, evidence_refs_education),
        
        QuestionEvaluation("DE-4-Q1", "¿Se evalúa la eficacia de programas educativos?", 
                          "No", 0.0, 0.2, 0.4, 0.0, evidence_refs_education[1:]),
        QuestionEvaluation("DE-4-Q2", "¿Las evaluaciones generan recomendaciones implementables?", 
                          "Parcial", 0.5, 0.5, 0.6, 0.50, evidence_refs_education)
    ]
    
    all_questions = health_questions + education_questions
    
    # Create dimension summaries
    dimensions_health = [
        DimensionSummary("DE-1", "Productos", health_questions[:3], 0.58, 3, 
                        {"lower": 0.45, "upper": 0.71}, {"equal_weight": 1.0}),
        DimensionSummary("DE-2", "Diagnóstico", health_questions[3:5], 0.82, 2,
                        {"lower": 0.68, "upper": 0.96}, {"equal_weight": 1.0}),
        DimensionSummary("DE-3", "Seguimiento", health_questions[5:7], 0.86, 2,
                        {"lower": 0.72, "upper": 1.0}, {"equal_weight": 1.0}),
        DimensionSummary("DE-4", "Evaluación", health_questions[7:9], 0.23, 2,
                        {"lower": 0.10, "upper": 0.36}, {"equal_weight": 1.0})
    ]
    
    dimensions_education = [
        DimensionSummary("DE-1", "Productos", education_questions[:2], 0.87, 2,
                        {"lower": 0.73, "upper": 1.01}, {"equal_weight": 1.0}),
        DimensionSummary("DE-2", "Diagnóstico", education_questions[2:4], 0.83, 2,
                        {"lower": 0.69, "upper": 0.97}, {"equal_weight": 1.0}),
        DimensionSummary("DE-3", "Seguimiento", education_questions[4:6], 0.81, 2,
                        {"lower": 0.67, "upper": 0.95}, {"equal_weight": 1.0}),
        DimensionSummary("DE-4", "Evaluación", education_questions[6:8], 0.25, 2,
                        {"lower": 0.12, "upper": 0.38}, {"equal_weight": 1.0})
    ]
    
    all_dimensions = dimensions_health + dimensions_education
    
    # Create point summaries
    points = [
        PointSummary(2, "Derecho a la salud", dimensions_health, 0.62, "Parcialmente Satisfactorio", 9, "CLUSTER-2"),
        PointSummary(3, "Derecho a la educación", dimensions_education, 0.69, "Parcialmente Satisfactorio", 8, "CLUSTER-2")
    ]
    
    # Create meso cluster analysis
    clusters = [
        MesoClusterAnalysis(
            "CLUSTER-2", "Derechos Sociales Fundamentales", [2, 3],
            {
                "internal_health_education": ["shared_infrastructure", "common_population_groups"],
                "external_cluster_1": ["protection_mechanisms"],
                "external_cluster_4": ["territorial_overlap"]
            },
            0.655,
            {"thematic_coherence": 0.89, "policy_alignment": 0.76, "evidence_consistency": 0.82},
            {"high_density": 12, "medium_density": 8, "low_density": 3}
        )
    ]
    
    # Create macro alignment
    macro = MacroAlignment(
        document_stem,
        0.655,  # Overall Decálogo score
        {"CLUSTER-1": 0.45, "CLUSTER-2": 0.655, "CLUSTER-3": 0.32, "CLUSTER-4": 0.71, "CLUSTER-5": 0.58},
        {"DE-1": 0.725, "DE-2": 0.825, "DE-3": 0.835, "DE-4": 0.240},
        {"territorial_coverage": 0.67, "population_coverage": 0.73, "evidence_quality": 0.81},
        {"Satisfactory": 0, "Parcialmente Satisfactorio": 2, "Insatisfactory": 0, "No Identificado": 0},
        ["Strengthen evaluation systems", "Improve DE-4 dimension across all points", "Enhance evidence collection"]
    )
    
    return all_questions, all_dimensions, points, clusters, macro


def demonstrate_integration():
    """Demonstrate integration with realistic Decálogo data."""
    print("=== Integration Demo: Comprehensive Artifact Generation ===\n")
    
    # Generate artifacts for multiple documents
    documents = ["ACANDI-CHOCO", "AGUSTIN-CODAZZI-CESAR", "ISTMINA-CHOCO"]
    
    generator = ArtifactGenerator()
    
    for doc_stem in documents:
        print(f"Processing document: {doc_stem}")
        print("-" * 50)
        
        # Create realistic evaluation data
        questions, dimensions, points, clusters, macro = create_realistic_decalogo_data(doc_stem)
        
        # Generate comprehensive artifacts
        artifacts = generator.generate_comprehensive_artifacts(
            doc_stem, questions, dimensions, points, clusters, macro
        )
        
        print(f"Generated {len(artifacts)} artifact types:")
        for artifact_type, filepath in artifacts.items():
            file_size = Path(filepath).stat().st_size
            print(f"  {artifact_type}: {filepath} ({file_size:,} bytes)")
        
        # Validate artifacts
        validation = generator.validate_artifacts(doc_stem)
        all_valid = all(validation.values())
        status = "✓ VALID" if all_valid else "✗ INVALID"
        print(f"Validation: {status}")
        
        print()
    
    # Discover all artifacts
    print("=== Artifact Discovery ===")
    discovered = generator.discover_artifacts()
    
    for doc_stem, artifact_types in discovered.items():
        if len(artifact_types) >= 5:  # Only show complete artifact sets
            print(f"{doc_stem}: {', '.join(artifact_types)}")
    
    print(f"\nTotal documents with complete artifact sets: {sum(1 for types in discovered.values() if len(types) >= 5)}")
    
    return True


def demonstrate_utf8_encoding():
    """Demonstrate UTF-8 encoding with special characters."""
    print("\n=== UTF-8 Encoding Demo ===")
    
    # Create data with Spanish characters and accents
    evidence_with_accents = [
        EvidenceReference("EV-UTF8-001", "documento_institucional", "p. 23", 
                         "Descripción con acentos: atención, evaluación, población", 0.87),
        EvidenceReference("EV-UTF8-002", "entrevista_comunidad", "p. 15", 
                         "Testimonio: 'La niñez necesita protección especial'", 0.92)
    ]
    
    question_with_spanish = QuestionEvaluation(
        "DE-1-Q-UTF8", "¿Se garantiza la atención integral a la población vulnerable?", 
        "Sí", 1.0, 0.9, 0.85, 1.08, evidence_with_accents
    )
    
    generator = ArtifactGenerator()
    generator.generate_question_artifacts("UTF8-DEMO", [question_with_spanish])
    
    # Read back to verify UTF-8 encoding
    with open("canonical_flow/analysis/UTF8-DEMO_questions.json", 'r', encoding='utf-8') as f:
        content = f.read()
        if "atención" in content and "población" in content:
            print("✓ UTF-8 encoding preserved special characters")
        else:
            print("✗ UTF-8 encoding failed")
    
    return True


if __name__ == "__main__":
    try:
        success = True
        
        success &= demonstrate_integration()
        success &= demonstrate_utf8_encoding()
        
        if success:
            print("\n✓ Integration demo completed successfully!")
            sys.exit(0)
        else:
            print("\n✗ Integration demo failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n✗ Demo error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
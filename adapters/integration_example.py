"""
Integration Example: Anti-Corruption Adapters in Practice

Shows how to integrate the adapter modules into an existing pipeline
to break circular dependencies and enforce proper separation.
"""

import logging
from typing import Any, Dict
from .retrieval_analysis_adapter import RetrievalAnalysisAdapter
from .import_blocker import ImportBlocker
from .lineage_tracker import LineageTracker


class PipelineOrchestrator:
    """
    Example pipeline orchestrator using anti-corruption adapters
    to enforce separation between retrieval and analysis phases.
    """
    
    def __init__(self):
        # Initialize tracking and blocking components
        self.lineage_tracker = LineageTracker(monitoring_endpoint="http://monitoring.example.com")
        self.import_blocker = ImportBlocker(self.lineage_tracker)
        
        # Initialize adapters for each phase boundary
        self.retrieval_analysis_adapter = RetrievalAnalysisAdapter("main_pipeline_adapter")
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("Pipeline orchestrator initialized with anti-corruption adapters")
    
    def process_document_query(self, query: str, document_sources: list) -> Dict[str, Any]:
        """
        Process a document query through the full pipeline using adapters
        """
        
        self.logger.info(f"Processing query: {query}")
        
        # Phase 1: Retrieval (simulated)
        retrieval_result = self._simulate_retrieval_phase(query, document_sources)
        
        # Phase 2: Adapter Translation
        analysis_input = self.retrieval_analysis_adapter.translate_retrieval_to_analysis(
            retrieval_output=retrieval_result,
            context={'query': query, 'sources': document_sources}
        )
        
        # Phase 3: Analysis (simulated)
        analysis_result = self._simulate_analysis_phase(analysis_input)
        
        # Generate final report
        return self._generate_pipeline_report(query, retrieval_result, analysis_input, analysis_result)
    
    def _simulate_retrieval_phase(self, query: str, sources: list) -> Dict[str, Any]:
        """Simulate retrieval phase operations"""
        
        # Track retrieval operation
        self.lineage_tracker.track_component_operation(
            component_id="hybrid_retrieval_engine",
            operation_type="document_search",
            input_schema="search_query",
            output_schema="retrieval_results",
            dependencies=["vector_index", "lexical_index", "embedding_model"]
        )
        
        # Simulate retrieval results
        return {
            'query_id': f"query_{hash(query) % 10000}",
            'retrieved_chunks': [
                {
                    'chunk_id': f'chunk_{i}',
                    'content': f'Retrieved content {i} for query: {query[:50]}...',
                    'source': sources[i % len(sources)] if sources else 'unknown',
                    'metadata': {'relevance_score': 0.9 - (i * 0.1), 'section': f'section_{i}'}
                }
                for i in range(min(3, len(sources) + 1))
            ],
            'similarity_scores': [0.95, 0.87, 0.82][:len(sources)+1],
            'retrieval_metadata': {
                'algorithm': 'hybrid_egw',
                'total_candidates': 1000,
                'execution_time_ms': 150,
                'model_version': '1.2.3'
            }
        }
    
    def _simulate_analysis_phase(self, analysis_input: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate analysis phase operations"""
        
        # Track analysis operation
        self.lineage_tracker.track_component_operation(
            component_id="dnp_compliance_analyzer",
            operation_type="compliance_analysis",
            input_schema="analysis_input",
            output_schema="compliance_report",
            dependencies=["main_pipeline_adapter", "dnp_standards_db"]
        )
        
        # Simulate analysis results
        chunks = analysis_input.get('document_chunks', [])
        context = analysis_input.get('context', {})
        
        compliance_score = 0.8  # Simulated
        issues_found = []
        
        # Simulate finding compliance issues
        for chunk in chunks:
            content = chunk.get('content', '').lower()
            if 'budget' not in content:
                issues_found.append("Missing budget allocation details")
            if 'participation' not in content:
                issues_found.append("Insufficient participatory planning evidence")
        
        return {
            'compliance_score': compliance_score,
            'is_compliant': compliance_score >= 0.7,
            'issues_found': issues_found,
            'recommendations': [
                "Include detailed budget breakdown",
                "Document community engagement process",
                "Align with constitutional requirements"
            ],
            'analysis_metadata': {
                'chunks_analyzed': len(chunks),
                'processing_time_ms': 300,
                'dnp_standards_version': '2024.1'
            }
        }
    
    def _generate_pipeline_report(
        self, 
        query: str, 
        retrieval_result: Dict[str, Any], 
        analysis_input: Dict[str, Any], 
        analysis_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive pipeline report"""
        
        # Get adapter statistics
        adapter_stats = self.retrieval_analysis_adapter.get_adapter_statistics()
        
        # Get lineage summary
        lineage_summary = self.lineage_tracker.get_system_lineage_summary()
        
        # Get import violation summary
        import_summary = self.import_blocker.get_violation_summary()
        
        return {
            'query': query,
            'pipeline_results': {
                'retrieval_phase': {
                    'chunks_retrieved': len(retrieval_result.get('retrieved_chunks', [])),
                    'execution_time_ms': retrieval_result.get('retrieval_metadata', {}).get('execution_time_ms', 0)
                },
                'analysis_phase': {
                    'compliance_score': analysis_result.get('compliance_score', 0),
                    'is_compliant': analysis_result.get('is_compliant', False),
                    'issues_count': len(analysis_result.get('issues_found', [])),
                    'execution_time_ms': analysis_result.get('analysis_metadata', {}).get('processing_time_ms', 0)
                }
            },
            'adapter_performance': {
                'translation_success_rate': adapter_stats['success_rate'],
                'schema_mismatches': adapter_stats['schema_mismatches'],
                'total_translations': adapter_stats['total_translations']
            },
            'dependency_health': {
                'system_health': lineage_summary['system_health'],
                'total_components': lineage_summary['total_components'],
                'circular_dependencies': lineage_summary['circular_dependencies'],
                'import_violations': import_summary['total_violations']
            },
            'recommendations': analysis_result.get('recommendations', []),
            'pipeline_metadata': {
                'phases_completed': 2,
                'adapter_used': True,
                'dependency_monitoring': True,
                'import_blocking_active': True
            }
        }
    
    def get_pipeline_health(self) -> Dict[str, Any]:
        """Get overall pipeline health assessment"""
        
        adapter_stats = self.retrieval_analysis_adapter.get_adapter_statistics()
        lineage_summary = self.lineage_tracker.get_system_lineage_summary()
        import_summary = self.import_blocker.get_violation_summary()
        
        # Calculate overall health score
        health_factors = {
            'adapter_success_rate': adapter_stats['success_rate'],
            'system_health': 1.0 if lineage_summary['system_health'] == 'healthy' else 0.5,
            'no_circular_deps': 1.0 if lineage_summary['circular_dependencies'] == 0 else 0.0,
            'no_import_violations': 1.0 if import_summary['total_violations'] == 0 else 0.7
        }
        
        overall_score = sum(health_factors.values()) / len(health_factors)
        
        return {
            'overall_health_score': overall_score,
            'health_status': self._get_health_status(overall_score),
            'health_factors': health_factors,
            'adapter_statistics': adapter_stats,
            'dependency_status': lineage_summary,
            'import_status': import_summary,
            'recommendations': self._get_health_recommendations(health_factors)
        }
    
    def _get_health_status(self, score: float) -> str:
        """Convert health score to status"""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.8:
            return "good"
        elif score >= 0.6:
            return "fair"
        elif score >= 0.4:
            return "poor"
        else:
            return "critical"
    
    def _get_health_recommendations(self, factors: Dict[str, float]) -> list:
        """Generate health improvement recommendations"""
        recommendations = []
        
        if factors['adapter_success_rate'] < 0.9:
            recommendations.append("Review adapter translations and fix schema mismatches")
        
        if factors['system_health'] < 1.0:
            recommendations.append("Address component dependency issues")
        
        if factors['no_circular_deps'] < 1.0:
            recommendations.append("Break circular dependencies between components")
        
        if factors['no_import_violations'] < 1.0:
            recommendations.append("Fix direct import violations, use adapters instead")
        
        return recommendations


def demo_integration():
    """Demonstrate the integrated pipeline with anti-corruption adapters"""
    
    print("=== Anti-Corruption Adapter Integration Demo ===\n")
    
    # Initialize pipeline
    pipeline = PipelineOrchestrator()
    
    # Process a sample query
    query = "Check DNP compliance for territorial development project"
    sources = ["project_pdt.pdf", "budget_report.xlsx", "community_input.docx"]
    
    print("1. Processing document query through adapter-protected pipeline...")
    result = pipeline.process_document_query(query, sources)
    
    print("✓ Query processed successfully")
    print(f"   Compliance score: {result['pipeline_results']['analysis_phase']['compliance_score']:.2f}")
    print(f"   Adapter success rate: {result['adapter_performance']['translation_success_rate']:.2%}")
    print(f"   System health: {result['dependency_health']['system_health']}")
    
    # Check pipeline health
    print("\n2. Checking overall pipeline health...")
    health = pipeline.get_pipeline_health()
    
    print(f"✓ Pipeline health assessment completed")
    print(f"   Overall health score: {health['overall_health_score']:.2f}")
    print(f"   Health status: {health['health_status']}")
    print(f"   Import violations: {health['import_status']['total_violations']}")
    
    if health['recommendations']:
        print(f"   Recommendations: {len(health['recommendations'])}")
        for rec in health['recommendations'][:3]:
            print(f"     - {rec}")
    
    print("\n✓ Integration demo completed successfully!")
    return pipeline


def run_integration_demo():
    """Run the integration demo - separate function for testing"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        pipeline = demo_integration()
        
        print("\n🎉 Integration successful!")
        print("\nAnti-corruption adapters provide:")
        print("- ✓ Clean separation between retrieval and analysis")
        print("- ✓ Automatic schema translation and validation")
        print("- ✓ Dependency violation monitoring")
        print("- ✓ Import blocking to prevent circular dependencies")
        print("- ✓ Comprehensive pipeline health monitoring")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_integration_demo()
    if not success:
        exit(1)
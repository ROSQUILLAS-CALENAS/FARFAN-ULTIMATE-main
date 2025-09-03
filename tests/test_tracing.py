"""
Tests for OpenTelemetry tracing system
"""

import time
import pytest
from unittest.mock import patch, MagicMock

from tracing import (
    PipelineTracer,
    get_pipeline_tracer,
    set_pipeline_tracer,
    trace_process,
    BackEdgeDetector,
    DependencyHeatmapVisualizer,
    TracingDashboard
)


class TestPipelineTracer:
    """Test pipeline tracer functionality"""
    
    def setup_method(self):
        """Set up test tracer"""
        self.tracer = PipelineTracer(service_name="test-pipeline")
        set_pipeline_tracer(self.tracer)
        
    def test_create_edge_span(self):
        """Test edge span creation"""
        span_id = self.tracer.create_edge_span(
            source_phase='I_ingestion_preparation',
            target_phase='X_context_construction',
            component_name='test_component',
            data={"test": "data"},
            context={"run_id": "test_001"}
        )
        
        assert span_id is not None
        assert span_id in self.tracer.active_spans
        
        span_info = self.tracer.active_spans[span_id]
        metadata = span_info['metadata']
        
        assert metadata.source_phase == 'I_ingestion_preparation'
        assert metadata.target_phase == 'X_context_construction'
        assert metadata.component_name == 'test_component'
        assert metadata.edge_type == 'forward'
        
    def test_complete_edge_span(self):
        """Test edge span completion"""
        span_id = self.tracer.create_edge_span(
            source_phase='I_ingestion_preparation',
            target_phase='X_context_construction',
            component_name='test_component'
        )
        
        time.sleep(0.01)
        
        self.tracer.complete_edge_span(
            span_id,
            result={"success": True}
        )
        
        assert span_id not in self.tracer.active_spans
        assert len(self.tracer.span_history) == 1
        assert len(self.tracer.phase_transitions) == 1
        
        span = self.tracer.span_history[0]
        assert span.timing_end is not None
        assert span.timing_end > span.timing_start
        
    def test_edge_classification(self):
        """Test edge type classification"""
        # Forward edge
        assert self.tracer._classify_edge(
            'I_ingestion_preparation', 
            'X_context_construction'
        ) == 'forward'
        
        # Backward edge (violation)
        assert self.tracer._classify_edge(
            'K_knowledge_extraction',
            'I_ingestion_preparation'
        ) == 'backward'
        
        # Self edge
        assert self.tracer._classify_edge(
            'A_analysis_nlp',
            'A_analysis_nlp'
        ) == 'self'
        
    def test_metrics_collection(self):
        """Test metrics collection"""
        # Create several spans
        for i in range(3):
            span_id = self.tracer.create_edge_span(
                source_phase='I_ingestion_preparation',
                target_phase='X_context_construction',
                component_name=f'component_{i}'
            )
            time.sleep(0.01)
            self.tracer.complete_edge_span(span_id)
            
        frequencies = self.tracer.get_edge_frequencies()
        assert frequencies['I_ingestion_preparation->X_context_construction'] == 3
        
        latencies = self.tracer.get_latency_distributions()
        assert len(latencies['I_ingestion_preparation->X_context_construction']) == 3


class TestTraceDecorator:
    """Test tracing decorators"""
    
    def setup_method(self):
        """Set up test tracer"""
        self.tracer = PipelineTracer(service_name="test-pipeline")
        set_pipeline_tracer(self.tracer)
        
    def test_trace_process_decorator(self):
        """Test @trace_process decorator"""
        
        @trace_process(
            source_phase='I_ingestion_preparation',
            target_phase='X_context_construction',
            component_name='test_function'
        )
        def test_function(data=None, context=None):
            time.sleep(0.01)
            return {"processed": True}
            
        result = test_function({"input": "test"}, {"run": "test"})
        
        assert result["processed"] is True
        assert len(self.tracer.span_history) == 1
        
        span = self.tracer.span_history[0]
        assert span.component_name == 'test_function'
        assert span.source_phase == 'I_ingestion_preparation'
        
    def test_trace_process_with_exception(self):
        """Test tracing with exceptions"""
        
        @trace_process(
            source_phase='I_ingestion_preparation',
            target_phase='X_context_construction',
            component_name='failing_function'
        )
        def failing_function(data=None, context=None):
            raise ValueError("Test error")
            
        with pytest.raises(ValueError):
            failing_function()
            
        assert len(self.tracer.span_history) == 1
        span = self.tracer.span_history[0]
        assert span.error is not None
        assert "Test error" in span.error


class TestBackEdgeDetector:
    """Test back-edge violation detection"""
    
    def setup_method(self):
        """Set up test environment"""
        self.tracer = PipelineTracer(service_name="test-pipeline")
        set_pipeline_tracer(self.tracer)
        self.detector = BackEdgeDetector()
        
    def test_detect_direct_back_edges(self):
        """Test direct backward edge detection"""
        # Create backward edge violation
        span_id = self.tracer.create_edge_span(
            source_phase='K_knowledge_extraction',
            target_phase='I_ingestion_preparation',  # Backward!
            component_name='violating_component'
        )
        self.tracer.complete_edge_span(span_id)
        
        violations = self.detector.analyze_span_traces()
        
        back_edge_violations = [
            v for v in violations 
            if v.violation_type == 'direct_back_edge'
        ]
        
        assert len(back_edge_violations) == 1
        violation = back_edge_violations[0]
        assert violation.source_phase == 'K_knowledge_extraction'
        assert violation.target_phase == 'I_ingestion_preparation'
        assert violation.severity == 'critical'
        
    def test_detect_phase_ordering_violations(self):
        """Test phase ordering violation detection"""
        # Create spans in wrong order
        span_id1 = self.tracer.create_edge_span(
            source_phase='L_classification_evaluation',
            target_phase='L_classification_evaluation',
            component_name='classifier'
        )
        time.sleep(0.01)
        self.tracer.complete_edge_span(span_id1)
        
        span_id2 = self.tracer.create_edge_span(
            source_phase='A_analysis_nlp',  # Should come before L
            target_phase='A_analysis_nlp',
            component_name='analyzer'
        )
        self.tracer.complete_edge_span(span_id2)
        
        violations = self.detector.analyze_span_traces()
        
        # Should detect ordering violation
        assert len(violations) > 0
        
    def test_violation_summary(self):
        """Test violation summary generation"""
        # Create some violations
        for i in range(3):
            span_id = self.tracer.create_edge_span(
                source_phase='R_search_retrieval',
                target_phase='K_knowledge_extraction',
                component_name=f'violating_component_{i}'
            )
            self.tracer.complete_edge_span(span_id)
            
        violations = self.detector.analyze_span_traces()
        summary = self.detector.get_violation_summary()
        
        assert summary['total_violations'] > 0
        assert 'by_type' in summary
        assert 'by_severity' in summary
        assert 'recent_violations' in summary


class TestDependencyHeatmapVisualizer:
    """Test heatmap visualization system"""
    
    def setup_method(self):
        """Set up test environment"""
        self.tracer = PipelineTracer(service_name="test-pipeline")
        set_pipeline_tracer(self.tracer)
        self.visualizer = DependencyHeatmapVisualizer()
        
    def test_generate_heatmap_data(self):
        """Test heatmap data generation"""
        # Create some sample spans
        for i in range(2):
            span_id = self.tracer.create_edge_span(
                source_phase='I_ingestion_preparation',
                target_phase='X_context_construction',
                component_name=f'component_{i}'
            )
            time.sleep(0.01)
            self.tracer.complete_edge_span(span_id)
            
        data = self.visualizer.generate_heatmap_data()
        
        assert 'timestamp' in data
        assert 'edge_frequencies' in data
        assert 'edge_latency_stats' in data
        assert 'phase_activity' in data
        assert 'canonical_phases' in data
        
        assert data['edge_frequencies']['I_ingestion_preparation->X_context_construction'] == 2
        
    def test_generate_html_dashboard(self):
        """Test HTML dashboard generation"""
        # Create sample data
        span_id = self.tracer.create_edge_span(
            source_phase='I_ingestion_preparation',
            target_phase='X_context_construction',
            component_name='test_component'
        )
        time.sleep(0.01)
        self.tracer.complete_edge_span(span_id)
        
        html = self.visualizer.generate_html_dashboard()
        
        assert isinstance(html, str)
        assert '<html>' in html
        assert 'Pipeline Dependency Heatmap' in html
        assert 'I_ingestion_preparation' in html
        assert 'X_context_construction' in html


class TestTracingDashboard:
    """Test tracing dashboard server"""
    
    def test_dashboard_creation(self):
        """Test dashboard creation"""
        dashboard = TracingDashboard(host='localhost', port=8081)
        
        assert dashboard.host == 'localhost'
        assert dashboard.port == 8081
        assert dashboard.visualizer is not None
        assert dashboard.detector is not None
        
    @patch('tracing.dashboard.HTTPServer')
    def test_dashboard_start_stop(self, mock_server):
        """Test dashboard start/stop"""
        dashboard = TracingDashboard(host='localhost', port=8081)
        
        # Mock server
        mock_server_instance = MagicMock()
        mock_server.return_value = mock_server_instance
        
        dashboard.start()
        
        assert dashboard.server is not None
        assert dashboard.server_thread is not None
        
        dashboard.stop()
        
        mock_server_instance.shutdown.assert_called_once()
        mock_server_instance.server_close.assert_called_once()


def test_integration():
    """Integration test of full tracing system"""
    # Set up tracer
    tracer = PipelineTracer(service_name="integration-test")
    set_pipeline_tracer(tracer)
    
    # Create traced function
    @trace_process(
        source_phase='I_ingestion_preparation',
        target_phase='X_context_construction',
        component_name='integration_test_component'
    )
    def test_process(data=None, context=None):
        time.sleep(0.01)
        return {"integration_test": True}
        
    # Execute
    result = test_process({"test": "data"}, {"test": "context"})
    
    # Verify tracing worked
    assert result["integration_test"] is True
    assert len(tracer.span_history) == 1
    
    # Test violation detection
    detector = BackEdgeDetector()
    violations = detector.analyze_span_traces()
    
    # Should be no violations for forward edge
    back_edges = [v for v in violations if v.violation_type == 'direct_back_edge']
    assert len(back_edges) == 0
    
    # Test visualization
    visualizer = DependencyHeatmapVisualizer()
    data = visualizer.generate_heatmap_data()
    
    assert data['total_spans'] == 1
    assert 'I_ingestion_preparation->X_context_construction' in data['edge_frequencies']
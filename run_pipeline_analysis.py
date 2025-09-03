#!/usr/bin/env python3
"""
Run Pipeline Analysis

Test script to run the pipeline value analysis system with the mock data.
"""

import json
# # # from pipeline_value_analysis_system import PipelineValueAnalysisSystem  # Module not found  # Module not found  # Module not found
# # # from stage_justification_framework import create_stage_justification_framework  # Module not found  # Module not found  # Module not found


def main():
    print('Testing Pipeline Value Analysis System')
    print('=' * 40)
    
    # Initialize the system
    analyzer = PipelineValueAnalysisSystem()
    
    # Run analysis
    print('Running stage analysis...')
    stage_analyses = analyzer.analyze_all_stages()
    print(f'✓ Analyzed {len(stage_analyses)} stages successfully')
    
    # Show stage results
    print('\nStage Analysis Results:')
    for name, analysis in stage_analyses.items():
        print(f'  {name}:')
        print(f'    Value: {analysis.value_contribution.value}')
        print(f'    Score: {analysis.justification_score:.3f}')
        print(f'    Issues: {len(analysis.issues)}')
        print(f'    Recommendations: {len(analysis.recommendations)}')
    
    # Generate report
    print('\nGenerating comprehensive report...')
    report = analyzer.generate_value_report()
    
    # Save report
    output_file = analyzer.save_analysis_report('test_pipeline_analysis_report.json')
    print(f'✓ Report saved to {output_file}')
    
    # Show summary
    print('\nSummary:')
    stats = report['summary_statistics']
    print(f'  Total stages: {stats["total_stages_analyzed"]}')
    print(f'  Average score: {stats["average_justification_score"]:.3f}')
    print(f'  Value distribution: {stats["value_distribution"]}')
    print(f'  Consolidation opportunities: {stats["total_consolidation_opportunities"]}')
    print(f'  Enhancement opportunities: {stats["total_enhancement_opportunities"]}')
    
    # Test stage justification framework
    print('\nTesting Stage Justification Framework...')
    stages_data = {}
    for stage_name, analysis in analyzer.stage_analyses.items():
        stages_data[stage_name] = {
            "stage_name": stage_name,
            "outputs": {k: str for k in analysis.artifact_profile.output_keys},
            "processing_metrics": {
                "execution_time": analysis.processing_metrics.execution_time,
                "transformation_ratio": analysis.processing_metrics.transformation_ratio,
                "computational_complexity": analysis.processing_metrics.computational_complexity
            },
            "artifact_profile": {
                "input_keys": analysis.artifact_profile.input_keys,
                "output_keys": analysis.artifact_profile.output_keys
            },
            "dependency_analysis": {
                "downstream_dependents": analysis.dependency_analysis.downstream_dependents,
                "removal_impact_score": analysis.dependency_analysis.removal_impact_score,
                "critical_path_position": analysis.dependency_analysis.critical_path_position
            }
        }
    
    framework = create_stage_justification_framework(stages_data)
    justification_report = framework.generate_justification_report()
    
    with open('test_stage_justification_report.json', 'w') as f:
        json.dump(justification_report, f, indent=2)
    
    print('✓ Stage justification report saved to test_stage_justification_report.json')
    
    # Show justification summary
    just_summary = justification_report["summary"]
    print(f'  Average justification score: {just_summary["average_justification_score"]:.3f}')
    print(f'  Justification levels: {just_summary["justification_levels"]}')
    print(f'  Total violations: {just_summary["total_threshold_violations"]}')
    
    print('\n✅ Pipeline Value Analysis System test completed successfully!')
    return True


if __name__ == "__main__":
    main()
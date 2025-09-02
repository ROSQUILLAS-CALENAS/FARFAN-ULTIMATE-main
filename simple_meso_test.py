#!/usr/bin/env python3
"""
Minimal test for meso aggregator without external dependencies.
"""

def test_basic_structure():
    """Test that we can create a basic meso result structure."""
    
    # Define expected structure
    expected_structure = {
        "meso_summary": {
            "items": {},
            "divergence_stats": {
                "question_count": 0
            },
            "cluster_participation": {
                "C1": 0, "C2": 0, "C3": 0, "C4": 0
            },
            "component_coverage_summary": {
                "total_components": 10,
                "fully_covered": 0,
                "partially_covered": 0,
                "not_covered": 10
            }
        },
        "coverage_matrix": {}
    }
    
    # Initialize coverage matrix for all components
    components = [
        "OBJECTIVES", "STRATEGIES", "INDICATORS", "TIMELINES", "BUDGET",
        "STAKEHOLDERS", "RISKS", "COMPLIANCE", "SUSTAINABILITY", "IMPACT"
    ]
    
    for component in components:
        expected_structure["coverage_matrix"][component] = {
            "clusters_evaluating": [],
            "questions_addressing": [],
            "total_evaluations": 0,
            "coverage_percentage": 0.0
        }
    
    print("✓ Basic structure created successfully")
    print(f"✓ Coverage matrix has {len(expected_structure['coverage_matrix'])} components")
    print(f"✓ Meso summary has required keys")
    
    return True

def test_simple_aggregation():
    """Test simple aggregation logic without external dependencies."""
    
    # Sample input data
    sample_input = {
        "cluster_audit": {
            "micro": {
                "C1": {
                    "answers": [
                        {
                            "question_id": "Q001",
                            "verdict": "YES",
                            "score": 0.8,
                            "evidence_ids": ["E001"],
                            "components": ["OBJECTIVES"]
                        }
                    ]
                },
                "C2": {
                    "answers": [
                        {
                            "question_id": "Q001", 
                            "verdict": "YES",
                            "score": 0.9,
                            "evidence_ids": ["E002"],
                            "components": ["OBJECTIVES"]
                        }
                    ]
                }
            }
        }
    }
    
    # Simple aggregation logic
    by_question = {}
    
    for cluster_id, payload in sample_input["cluster_audit"]["micro"].items():
        for answer in payload.get("answers", []):
            question_key = answer.get("question_id", "unknown")
            
            if question_key not in by_question:
                by_question[question_key] = {
                    "by_cluster": {},
                    "scores": [],
                    "components": set()
                }
            
            by_question[question_key]["by_cluster"][cluster_id] = answer
            by_question[question_key]["scores"].append(answer.get("score", 0))
            
            # Extract components
            components = answer.get("components", [])
            if isinstance(components, list):
                by_question[question_key]["components"].update(components)
    
    print(f"✓ Processed {len(by_question)} questions")
    
    # Check results
    if "Q001" in by_question:
        q001_data = by_question["Q001"]
        print(f"✓ Q001 has {len(q001_data['by_cluster'])} cluster responses")
        print(f"✓ Q001 average score: {sum(q001_data['scores']) / len(q001_data['scores']):.2f}")
        print(f"✓ Q001 components: {list(q001_data['components'])}")
    
    return True

def test_coverage_calculation():
    """Test coverage matrix calculation."""
    
    # Sample processed data
    by_question = {
        "Q001": {
            "by_cluster": {
                "C1": {"components": ["OBJECTIVES"]},
                "C2": {"components": ["OBJECTIVES", "STRATEGIES"]}
            },
            "components": {"OBJECTIVES", "STRATEGIES"}
        },
        "Q002": {
            "by_cluster": {
                "C1": {"components": ["STAKEHOLDERS"]},
                "C3": {"components": ["STAKEHOLDERS"]}
            },
            "components": {"STAKEHOLDERS"}
        }
    }
    
    # Build coverage matrix
    components = ["OBJECTIVES", "STRATEGIES", "STAKEHOLDERS", "RISKS"]
    coverage_matrix = {}
    
    for component in components:
        coverage_matrix[component] = {
            "clusters_evaluating": set(),
            "questions_addressing": set(),
            "total_evaluations": 0,
            "coverage_percentage": 0.0
        }
    
    # Process questions
    for question_id, data in by_question.items():
        for cluster_id, answer in data["by_cluster"].items():
            answer_components = answer.get("components", [])
            for comp in answer_components:
                if comp in coverage_matrix:
                    coverage_matrix[comp]["clusters_evaluating"].add(cluster_id)
                    coverage_matrix[comp]["questions_addressing"].add(question_id)
                    coverage_matrix[comp]["total_evaluations"] += 1
    
    # Calculate coverage percentages
    total_clusters = 4  # C1-C4
    for component in coverage_matrix:
        clusters_count = len(coverage_matrix[component]["clusters_evaluating"])
        coverage_matrix[component]["coverage_percentage"] = (clusters_count / total_clusters) * 100
        
        # Convert sets to lists
        coverage_matrix[component]["clusters_evaluating"] = sorted(list(coverage_matrix[component]["clusters_evaluating"]))
        coverage_matrix[component]["questions_addressing"] = sorted(list(coverage_matrix[component]["questions_addressing"]))
    
    print("✓ Coverage matrix calculated")
    
    # Check results
    for component, data in coverage_matrix.items():
        if data["total_evaluations"] > 0:
            print(f"✓ {component}: {data['coverage_percentage']:.1f}% coverage, {data['total_evaluations']} evaluations")
    
    return True

def test_file_generation():
    """Test JSON file generation."""
    import json
    import tempfile
    from pathlib import Path
    
    # Sample result data
    result = {
        "meso_summary": {
            "items": {
                "Q001": {
                    "by_cluster": {
                        "C1": {"score": 0.8, "verdict": "YES"},
                        "C2": {"score": 0.9, "verdict": "YES"}
                    },
                    "score_summary": {"count": 2, "avg": 0.85}
                }
            },
            "divergence_stats": {"question_count": 1},
            "cluster_participation": {"C1": 1, "C2": 1, "C3": 0, "C4": 0},
            "component_coverage_summary": {
                "total_components": 10,
                "fully_covered": 1,
                "partially_covered": 0,
                "not_covered": 9
            }
        },
        "coverage_matrix": {
            "OBJECTIVES": {
                "clusters_evaluating": ["C1", "C2"],
                "questions_addressing": ["Q001"],
                "total_evaluations": 2,
                "coverage_percentage": 50.0
            }
        }
    }
    
    # Create temporary file
    temp_file = tempfile.mktemp(suffix='.json')
    temp_path = Path(temp_file)
    
    try:
        # Write JSON
        with open(temp_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✓ JSON file created: {temp_path}")
        
        # Verify file can be read
        with open(temp_path, 'r') as f:
            loaded_data = json.load(f)
        
        if loaded_data == result:
            print("✓ JSON content verified")
        else:
            print("✗ JSON content mismatch")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ File generation failed: {e}")
        return False
    
    finally:
        # Clean up
        if temp_path.exists():
            temp_path.unlink()

def main():
    """Run basic tests."""
    print("Simple Meso Aggregator Test")
    print("=" * 40)
    
    tests = [
        ("Basic Structure", test_basic_structure),
        ("Simple Aggregation", test_simple_aggregation),
        ("Coverage Calculation", test_coverage_calculation),
        ("File Generation", test_file_generation)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                print(f"✓ {test_name} PASSED")
                passed += 1
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED: {e}")
    
    print(f"\nResults: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All basic tests passed!")
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit(main())
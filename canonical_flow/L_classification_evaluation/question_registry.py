"""
Decálogo Question Registry
Manages question definitions, validation rules, and checksum computation
for ensuring schema-data consistency across classification components.
"""

import json
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, List, Any, Optional, Set  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found

# # # from schemas import (  # Module not found  # Module not found  # Module not found
    RegistryChecksumValidator, 
    DimensionType, 
    ResponseType,
    QuestionEvalInput
)


@dataclass
class QuestionDefinition:
    """Definition of a single evaluation question"""
    question_id: str
    dimension: str
    text: str
    category: str
    weight: float = 1.0
    required: bool = True
    validation_rules: List[str] = None
    
    def __post_init__(self):
        if self.validation_rules is None:
            self.validation_rules = []


@dataclass 
class DimensionDefinition:
    """Definition of a Decálogo dimension"""
    dimension_id: str
    name: str
    description: str
    weight: float
    expected_question_count: int
    questions: List[QuestionDefinition]


class DecalogoQuestionRegistry:
    """
    Registry for Decálogo evaluation questions with checksum-based validation.
    Ensures consistency between question definitions and classification components.
    """
    
    def __init__(self, registry_file: Optional[str] = None):
        """
        Initialize question registry.
        
        Args:
            registry_file: Path to registry JSON file, if None uses default questions
        """
        self.registry_file = registry_file
        self.dimensions: Dict[str, DimensionDefinition] = {}
        self.questions: Dict[str, QuestionDefinition] = {}
        self.registry_metadata = {
            "version": "1.0.0",
            "created": datetime.utcnow().isoformat(),
            "last_modified": datetime.utcnow().isoformat(),
            "total_questions": 0,
            "checksum": ""
        }
        
# # #         # Load questions from file or initialize defaults  # Module not found  # Module not found  # Module not found
        if registry_file and Path(registry_file).exists():
            self.load_from_file(registry_file)
        else:
            self._initialize_default_questions()
        
        # Compute initial checksum
        self.update_checksum()
    
    def _initialize_default_questions(self) -> None:
        """Initialize default Decálogo question set"""
        # Define default dimension structure
        dimension_configs = [
            {
                "dimension_id": "DE-1",
                "name": "Dimensión Institucional", 
                "description": "Capacidad institucional y governance",
                "weight": 0.30,
                "expected_question_count": 12
            },
            {
                "dimension_id": "DE-2", 
                "name": "Dimensión Social",
                "description": "Desarrollo social y equidad",
                "weight": 0.25,
                "expected_question_count": 12
            },
            {
                "dimension_id": "DE-3",
                "name": "Dimensión Económica", 
                "description": "Desarrollo económico sostenible",
                "weight": 0.25,
                "expected_question_count": 12
            },
            {
                "dimension_id": "DE-4",
                "name": "Dimensión Ambiental",
                "description": "Sostenibilidad ambiental y territorial",
                "weight": 0.20, 
                "expected_question_count": 11
            }
        ]
        
        # Generate sample questions for each dimension
        for dim_config in dimension_configs:
            questions = []
            for i in range(1, dim_config["expected_question_count"] + 1):
                question = QuestionDefinition(
                    question_id=f"{dim_config['dimension_id'].replace('-', '')}_Q{i}",
                    dimension=dim_config["dimension_id"],
                    text=f"¿El PDT cumple con el criterio {i} de {dim_config['name']}?",
                    category="compliance",
                    weight=1.0,
                    required=True,
                    validation_rules=["requires_evidence", "requires_citation"]
                )
                questions.append(question)
                self.questions[question.question_id] = question
            
            dimension = DimensionDefinition(
                dimension_id=dim_config["dimension_id"],
                name=dim_config["name"],
                description=dim_config["description"], 
                weight=dim_config["weight"],
                expected_question_count=dim_config["expected_question_count"],
                questions=questions
            )
            self.dimensions[dimension.dimension_id] = dimension
        
        # Update metadata
        self.registry_metadata["total_questions"] = len(self.questions)
        self.registry_metadata["last_modified"] = datetime.utcnow().isoformat()
    
    def load_from_file(self, file_path: str) -> None:
# # #         """Load registry from JSON file"""  # Module not found  # Module not found  # Module not found
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load metadata
            self.registry_metadata = data.get("metadata", self.registry_metadata)
            
            # Load dimensions and questions
            self.dimensions = {}
            self.questions = {}
            
            for dim_data in data.get("dimensions", []):
                questions = []
                for q_data in dim_data.get("questions", []):
                    question = QuestionDefinition(
                        question_id=q_data["question_id"],
                        dimension=q_data["dimension"],
                        text=q_data["text"],
                        category=q_data.get("category", "compliance"),
                        weight=q_data.get("weight", 1.0),
                        required=q_data.get("required", True),
                        validation_rules=q_data.get("validation_rules", [])
                    )
                    questions.append(question)
                    self.questions[question.question_id] = question
                
                dimension = DimensionDefinition(
                    dimension_id=dim_data["dimension_id"],
                    name=dim_data["name"],
                    description=dim_data["description"],
                    weight=dim_data["weight"],
                    expected_question_count=dim_data["expected_question_count"],
                    questions=questions
                )
                self.dimensions[dimension.dimension_id] = dimension
                
        except Exception as e:
# # #             raise RuntimeError(f"Failed to load registry from {file_path}: {e}")  # Module not found  # Module not found  # Module not found
    
    def save_to_file(self, file_path: str) -> None:
        """Save registry to JSON file"""
        try:
            # Build export data structure
            export_data = {
                "metadata": self.registry_metadata,
                "dimensions": []
            }
            
            for dimension in self.dimensions.values():
                dim_data = {
                    "dimension_id": dimension.dimension_id,
                    "name": dimension.name,
                    "description": dimension.description,
                    "weight": dimension.weight,
                    "expected_question_count": dimension.expected_question_count,
                    "questions": []
                }
                
                for question in dimension.questions:
                    q_data = {
                        "question_id": question.question_id,
                        "dimension": question.dimension,
                        "text": question.text,
                        "category": question.category,
                        "weight": question.weight,
                        "required": question.required,
                        "validation_rules": question.validation_rules
                    }
                    dim_data["questions"].append(q_data)
                
                export_data["dimensions"].append(dim_data)
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            raise RuntimeError(f"Failed to save registry to {file_path}: {e}")
    
    def get_registry_data(self) -> Dict[str, Any]:
        """Get complete registry data for checksum computation"""
        return {
            "metadata": self.registry_metadata,
            "dimensions": {
                dim_id: {
                    "name": dim.name,
                    "description": dim.description,
                    "weight": dim.weight,
                    "expected_question_count": dim.expected_question_count,
                    "questions": {
                        q.question_id: {
                            "text": q.text,
                            "category": q.category,
                            "weight": q.weight,
                            "required": q.required,
                            "validation_rules": q.validation_rules
                        } for q in dim.questions
                    }
                } for dim_id, dim in self.dimensions.items()
            }
        }
    
    def update_checksum(self) -> str:
        """Update and return registry checksum"""
        registry_data = self.get_registry_data()
        checksum = RegistryChecksumValidator.compute_registry_checksum(registry_data)
        self.registry_metadata["checksum"] = checksum
        self.registry_metadata["last_modified"] = datetime.utcnow().isoformat()
        return checksum
    
    def get_checksum(self) -> str:
        """Get current registry checksum"""
        return self.registry_metadata["checksum"]
    
    def validate_checksum(self, expected_checksum: str) -> bool:
        """Validate current registry against expected checksum"""
        registry_data = self.get_registry_data()
        return RegistryChecksumValidator.validate_registry_checksum(
            expected_checksum, registry_data
        )
    
    def get_question(self, question_id: str) -> Optional[QuestionDefinition]:
        """Get question by ID"""
        return self.questions.get(question_id)
    
    def get_dimension_questions(self, dimension_id: str) -> List[QuestionDefinition]:
        """Get all questions for a dimension"""
        dimension = self.dimensions.get(dimension_id)
        return dimension.questions if dimension else []
    
    def get_dimension(self, dimension_id: str) -> Optional[DimensionDefinition]:
        """Get dimension definition by ID"""
        return self.dimensions.get(dimension_id)
    
    def validate_question_input(self, question_data: Dict[str, Any]) -> bool:
        """
        Validate question input data against registry definitions.
        
        Args:
            question_data: Question data dict to validate
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            # Validate against Pydantic schema first
            validated_input = QuestionEvalInput(**question_data)
            
            # Validate question exists in registry
            question = self.get_question(validated_input.question_id)
            if not question:
                return False
            
            # Validate dimension alignment
            expected_dimension = question.dimension
            if validated_input.dimension.value != expected_dimension:
                return False
            
            # Validate registry checksum if provided
            if hasattr(validated_input, 'registry_checksum'):
                if not self.validate_checksum(validated_input.registry_checksum):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def get_dimension_weights(self) -> Dict[str, float]:
        """Get dimension weights for scoring calculations"""
        return {
            dim_id: dim.weight 
            for dim_id, dim in self.dimensions.items()
        }
    
    def get_question_count_by_dimension(self) -> Dict[str, int]:
        """Get question counts for each dimension"""
        return {
            dim_id: len(dim.questions)
            for dim_id, dim in self.dimensions.items() 
        }
    
    def get_total_question_count(self) -> int:
        """Get total number of questions across all dimensions"""
        return sum(len(dim.questions) for dim in self.dimensions.values())
    
    def validate_evaluation_completeness(
        self, 
        evaluation_data: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Validate evaluation data completeness against registry expectations.
        
        Args:
            evaluation_data: Evaluation data by dimension
            
        Returns:
            Validation report with completeness metrics
        """
        report = {
            "is_complete": True,
            "total_expected": self.get_total_question_count(),
            "total_provided": 0,
            "dimension_completeness": {},
            "missing_questions": [],
            "unexpected_questions": []
        }
        
        # Check each dimension
        for dim_id, dim in self.dimensions.items():
            expected_questions = {q.question_id for q in dim.questions}
            provided_data = evaluation_data.get(dim_id, [])
            provided_questions = {item.get("question_id") for item in provided_data if item.get("question_id")}
            
            missing = expected_questions - provided_questions
            unexpected = provided_questions - expected_questions
            
            report["dimension_completeness"][dim_id] = {
                "expected": len(expected_questions),
                "provided": len(provided_questions),
                "missing": list(missing),
                "unexpected": list(unexpected),
                "completeness_ratio": len(provided_questions) / len(expected_questions) if expected_questions else 1.0
            }
            
            report["total_provided"] += len(provided_questions)
            report["missing_questions"].extend(missing)
            report["unexpected_questions"].extend(unexpected)
            
            if missing or unexpected:
                report["is_complete"] = False
        
        report["overall_completeness_ratio"] = (
            report["total_provided"] / report["total_expected"] 
            if report["total_expected"] > 0 else 1.0
        )
        
        return report
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            "version": self.registry_metadata["version"],
            "total_questions": len(self.questions),
            "total_dimensions": len(self.dimensions),
            "dimension_breakdown": self.get_question_count_by_dimension(),
            "dimension_weights": self.get_dimension_weights(),
            "last_modified": self.registry_metadata["last_modified"],
            "checksum": self.get_checksum()
        }


# Global registry instance (singleton pattern)
_global_registry: Optional[DecalogoQuestionRegistry] = None


def get_default_registry() -> DecalogoQuestionRegistry:
    """Get the default global registry instance"""
    global _global_registry
    if _global_registry is None:
        _global_registry = DecalogoQuestionRegistry()
    return _global_registry


def create_custom_registry(registry_file: str) -> DecalogoQuestionRegistry:
# # #     """Create a custom registry from file"""  # Module not found  # Module not found  # Module not found
    return DecalogoQuestionRegistry(registry_file)
"""
PDT Validator Implementation

Concrete implementation of ValidatorPort for PDT document validation.
Only depends on validator_api interfaces.
"""

from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass

# Import only from validator_api - no pipeline dependencies
from validator_api.validation_interfaces import ValidatorPort, ValidationResult, ValidationStatus
from validator_api.dtos import ValidationRequest


class SectionType(Enum):
    """Types of sections in PDT documents."""
    DIAGNOSTICO = "diagnostico"
    PROGRAMAS = "programas"
    PRESUPUESTO = "presupuesto"
    METAS = "metas"
    SEGUIMIENTO = "seguimiento"


@dataclass
class SectionBlock:
    """Represents a section block in a PDT document."""
    section_type: SectionType
    text: str
    confidence: float
    page_number: Optional[int] = None
    position: Optional[int] = None


@dataclass
class QualityIndicators:
    """Quality indicators for PDT documents."""
    completeness_index: float
    logical_coherence_hint: float
    tables_found: int
    ocr_ratio: float
    mandatory_sections_present: List[str]
    missing_sections: List[str]


class PDTValidator(ValidatorPort):
    """Validates PDT documents for compliance with DNP standards."""
    
    def __init__(self):
        # Mandatory sections according to DNP standards
        self.mandatory_sections = {
            SectionType.DIAGNOSTICO: {
                "required": True,
                "min_tokens": 1000,
                "description": "Diagnóstico situacional"
            },
            SectionType.PROGRAMAS: {
                "required": True,
                "min_tokens": 800,
                "description": "Programas y proyectos"
            },
            SectionType.PRESUPUESTO: {
                "required": True,
                "min_tokens": 500,
                "description": "Marco presupuestal"
            },
            SectionType.METAS: {
                "required": True,
                "min_tokens": 400,
                "description": "Metas e indicadores"
            },
            SectionType.SEGUIMIENTO: {
                "required": False,
                "min_tokens": 300,
                "description": "Sistema de seguimiento"
            }
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            "min_completeness": 0.8,
            "min_content_length": 1000,
            "max_ocr_ratio": 0.4,
            "min_coherence": 0.6
        }
        
        # Error handling policies
        self.error_handling = {
            "missing_mandatory": "flag_and_continue",
            "insufficient_length": "warning",
            "low_quality": "warning",
            "structural_issues": "flag_and_continue"
        }
    
    def validate(self, request: ValidationRequest) -> ValidationResult:
        """
        Validate PDT document structure and content.
        
        Args:
            request: Validation request containing PDT data
            
        Returns:
            ValidationResult with detailed validation status
        """
        try:
            # Extract blocks from request data
            blocks = self._extract_blocks_from_request(request)
            
            # Check mandatory sections
            section_validation = self._check_mandatory_sections(blocks)
            
            # Calculate quality indicators if metadata available
            quality_indicators = self._calculate_quality_indicators(
                blocks,
                request.context.get("ocr_pages", 0) if request.context else 0,
                request.context.get("total_pages", 1) if request.context else 1,
                request.context.get("tables_found", 0) if request.context else 0
            )
            
            # Apply error policies
            final_result = self._apply_error_policies(section_validation)
            
            # Determine overall status
            status = ValidationStatus.PASSED
            if section_validation["validation_result"] == "failed":
                status = ValidationStatus.FAILED
            elif section_validation["validation_result"] == "warning":
                status = ValidationStatus.WARNING
            
            return ValidationResult(
                status=status,
                message=self._generate_summary_message(section_validation, quality_indicators),
                details={
                    "section_validation": section_validation,
                    "quality_indicators": quality_indicators.__dict__,
                    "policies_applied": final_result
                },
                errors=section_validation.get("errors", []),
                warnings=section_validation.get("warnings", []),
                confidence_score=quality_indicators.completeness_index,
                metadata={
                    "validator_type": "PDT",
                    "standards_version": "DNP_2024",
                    "sections_analyzed": len(blocks)
                }
            )
            
        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.ERROR,
                message=f"Validation failed with error: {str(e)}",
                details={"error_details": str(e)},
                errors=[str(e)],
                warnings=[],
                confidence_score=0.0
            )
    
    def get_validation_rules(self) -> Dict[str, Any]:
        """Get the validation rules used by this validator."""
        return {
            "mandatory_sections": self.mandatory_sections,
            "quality_thresholds": self.quality_thresholds,
            "error_handling": self.error_handling
        }
    
    def supports_data_type(self, data_type: str) -> bool:
        """Check if this validator supports a given data type."""
        supported_types = ["pdt", "document", "pdf_document", "structured_document"]
        return data_type.lower() in supported_types
    
    def _extract_blocks_from_request(self, request: ValidationRequest) -> List[SectionBlock]:
        """Extract section blocks from validation request data."""
        blocks = []
        
        if isinstance(request.data, dict):
            # Handle structured data with blocks
            if "blocks" in request.data:
                for block_data in request.data["blocks"]:
                    blocks.append(self._parse_block_data(block_data))
            # Handle simple text data
            elif "text" in request.data:
                # Create a single block for simple text
                blocks.append(SectionBlock(
                    section_type=SectionType.DIAGNOSTICO,  # Default
                    text=request.data["text"],
                    confidence=0.8
                ))
        elif isinstance(request.data, list):
            # Handle list of blocks
            for block_data in request.data:
                blocks.append(self._parse_block_data(block_data))
        elif isinstance(request.data, str):
            # Handle plain text
            blocks.append(SectionBlock(
                section_type=SectionType.DIAGNOSTICO,  # Default
                text=request.data,
                confidence=0.8
            ))
        
        return blocks
    
    def _parse_block_data(self, block_data: Dict[str, Any]) -> SectionBlock:
        """Parse block data into SectionBlock object."""
        # Determine section type
        section_type_str = block_data.get("section_type", "diagnostico").lower()
        try:
            section_type = SectionType(section_type_str)
        except ValueError:
            section_type = SectionType.DIAGNOSTICO
        
        return SectionBlock(
            section_type=section_type,
            text=block_data.get("text", ""),
            confidence=block_data.get("confidence", 0.5),
            page_number=block_data.get("page_number"),
            position=block_data.get("position")
        )
    
    def _check_mandatory_sections(self, blocks: List[SectionBlock]) -> Dict[str, Any]:
        """Check presence and quality of mandatory sections."""
        result = {
            "validation_result": "passed",
            "missing_sections": [],
            "insufficient_sections": [],
            "present_sections": [],
            "warnings": [],
            "errors": []
        }
        
        # Map found sections
        found_sections = {block.section_type for block in blocks}
        
        # Check mandatory sections
        for section_type, config in self.mandatory_sections.items():
            if config["required"]:
                if section_type not in found_sections:
                    result["missing_sections"].append(section_type.value)
                    result["errors"].append(f"Missing mandatory section: {config['description']}")
                    result["validation_result"] = "failed"
                else:
                    result["present_sections"].append(section_type.value)
        
        # Check content length
        for block in blocks:
            section_config = self.mandatory_sections.get(block.section_type)
            if section_config:
                token_count = len(block.text.split())
                min_tokens = section_config["min_tokens"]
                
                if token_count < min_tokens:
                    result["insufficient_sections"].append({
                        "section": block.section_type.value,
                        "current_tokens": token_count,
                        "required_tokens": min_tokens
                    })
                    result["warnings"].append(
                        f"Section {block.section_type.value} has {token_count} tokens, "
                        f"requires minimum {min_tokens}"
                    )
                    if result["validation_result"] == "passed":
                        result["validation_result"] = "warning"
        
        return result
    
    def _calculate_quality_indicators(self, 
                                    blocks: List[SectionBlock],
                                    ocr_pages: int,
                                    total_pages: int,
                                    tables_found: int) -> QualityIndicators:
        """Calculate quality indicators for the document."""
        # Completeness index
        mandatory_count = sum(1 for st, cfg in self.mandatory_sections.items() if cfg["required"])
        found_mandatory = sum(1 for block in blocks 
                            if block.section_type in self.mandatory_sections 
                            and self.mandatory_sections[block.section_type]["required"])
        
        completeness_index = found_mandatory / mandatory_count if mandatory_count > 0 else 0.0
        
        # OCR ratio
        ocr_ratio = ocr_pages / total_pages if total_pages > 0 else 0.0
        
        # Logical coherence hint
        logical_coherence_hint = self._calculate_coherence_hint(blocks)
        
        # Sections info
        found_sections = [block.section_type.value for block in blocks]
        mandatory_sections_list = [st.value for st, cfg in self.mandatory_sections.items() if cfg["required"]]
        missing_sections = [sec for sec in mandatory_sections_list if sec not in found_sections]
        
        return QualityIndicators(
            completeness_index=completeness_index,
            logical_coherence_hint=logical_coherence_hint,
            tables_found=tables_found,
            ocr_ratio=ocr_ratio,
            mandatory_sections_present=found_sections,
            missing_sections=missing_sections
        )
    
    def _apply_error_policies(self, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply error handling policies to validation results."""
        processed_result = validation_result.copy()
        
        # Apply policy for missing mandatory sections
        if validation_result["missing_sections"]:
            if self.error_handling["missing_mandatory"] == "flag_and_continue":
                processed_result["continue_processing"] = True
                processed_result["flags"] = processed_result.get("flags", [])
                processed_result["flags"].append("missing_mandatory_sections")
                processed_result["confidence_reduction"] = 0.3
        
        # Apply policy for insufficient content
        if validation_result["insufficient_sections"]:
            if self.error_handling["insufficient_length"] == "warning":
                # Already handled as warning
                pass
        
        return processed_result
    
    def _calculate_coherence_hint(self, blocks: List[SectionBlock]) -> float:
        """Calculate basic coherence hint for document structure."""
        if not blocks:
            return 0.0
        
        coherence_score = 0.0
        factors = []
        
        # Factor 1: Logical order of sections
        expected_order = [
            SectionType.DIAGNOSTICO,
            SectionType.PROGRAMAS,
            SectionType.PRESUPUESTO,
            SectionType.METAS,
            SectionType.SEGUIMIENTO
        ]
        
        section_positions = {}
        for i, block in enumerate(blocks):
            if block.section_type not in section_positions:
                section_positions[block.section_type] = i
        
        order_violations = 0
        for i in range(len(expected_order) - 1):
            current_type = expected_order[i]
            next_type = expected_order[i + 1]
            
            if (current_type in section_positions and 
                next_type in section_positions and
                section_positions[current_type] > section_positions[next_type]):
                order_violations += 1
        
        order_factor = max(0.0, 1.0 - (order_violations / max(1, len(expected_order) - 1)))
        factors.append(order_factor)
        
        # Factor 2: Content distribution
        total_text = sum(len(block.text) for block in blocks)
        if total_text > 0:
            max_section_ratio = max(len(block.text) / total_text for block in blocks)
            balance_factor = 1.0 if max_section_ratio < 0.6 else (1.0 - max_section_ratio)
            factors.append(balance_factor)
        
        # Factor 3: Average confidence
        if blocks:
            avg_confidence = sum(block.confidence for block in blocks) / len(blocks)
            factors.append(avg_confidence)
        
        # Calculate average
        coherence_score = sum(factors) / len(factors) if factors else 0.0
        
        return min(1.0, max(0.0, coherence_score))
    
    def _generate_summary_message(self, section_validation: Dict[str, Any], quality_indicators: QualityIndicators) -> str:
        """Generate a summary message for validation results."""
        status = section_validation["validation_result"]
        
        if status == "passed":
            return f"PDT validation passed. Completeness: {quality_indicators.completeness_index:.2%}, " \
                   f"Coherence: {quality_indicators.logical_coherence_hint:.2%}"
        elif status == "warning":
            warning_count = len(section_validation.get("warnings", []))
            return f"PDT validation passed with {warning_count} warnings. " \
                   f"Completeness: {quality_indicators.completeness_index:.2%}"
        else:
            error_count = len(section_validation.get("errors", []))
            missing_count = len(section_validation.get("missing_sections", []))
            return f"PDT validation failed with {error_count} errors. " \
                   f"Missing {missing_count} mandatory sections."
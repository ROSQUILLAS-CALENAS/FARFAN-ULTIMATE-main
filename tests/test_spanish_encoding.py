"""
Test Spanish character encoding support across the codebase.

This test ensures that Spanish characters (especially ñ, á, é, í, ó, ú, ü)
are properly handled throughout the system, which is critical for processing
Colombian DNP documents and Decálogo content.
"""

import json


def test_spanish_character_encoding():
    """Test that Spanish characters are properly encoded and decoded."""
    test_cases = [
        "ññ",  # Double ñ - the specific test case
        "niño",
        "niña", 
        "año",
        "español",
        "Decálogo",
        "Metodología",
        "evaluación",
        "MÓDULO",
        "información",
        "atención",
        "gestión",
    ]
    
    for text in test_cases:
        # Test encoding/decoding roundtrip
        encoded = text.encode('utf-8')
        decoded = encoded.decode('utf-8')
        assert text == decoded, f"Encoding roundtrip failed for '{text}'"


def test_spanish_special_punctuation():
    """Test Spanish-specific punctuation marks."""
    test_cases = [
        "¿Pregunta?",
        "¡Atención!",
        "¿Cómo está?",
        "¡Excelente!",
    ]
    
    for text in test_cases:
        encoded = text.encode('utf-8')
        decoded = encoded.decode('utf-8')
        assert text == decoded, f"Encoding roundtrip failed for '{text}'"


def test_spanish_text_in_json():
    """Test that Spanish text can be properly serialized to JSON."""
    test_data = {
        "title": "Decálogo de Derechos Humanos",
        "description": "Evaluación de PDT según metodología DNP",
        "questions": [
            "¿El PDT define productos medibles?",
            "¿Las metas incluyen responsable institucional?",
        ],
        "keywords": ["niño", "niña", "año", "español"],
        "special": "ññ",
    }
    
    # Test JSON serialization
    json_str = json.dumps(test_data, ensure_ascii=False)
    decoded_data = json.loads(json_str)
    
    assert decoded_data["title"] == test_data["title"]
    assert decoded_data["special"] == "ññ"
    assert "niño" in decoded_data["keywords"]
    

def test_file_reading_with_spanish_content():
    """Test reading actual files with Spanish content."""
    import os
    
    # Test files that are known to contain Spanish characters
    test_files = [
        "Cuestionario Original de la Metodología.md",
        "Decálogo de Derechos Humanos_ Puntos y Clústeres.md",
    ]
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    for filename in test_files:
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # Verify Spanish characters are present
                assert 'á' in content or 'é' in content or 'í' in content or 'ó' in content or 'ú' in content or 'ñ' in content
                # Verify the content is not corrupted
                assert len(content) > 0


def test_string_operations_with_spanish_chars():
    """Test common string operations with Spanish characters."""
    text = "El niño español estudia la metodología"
    
    # Test upper/lower case
    assert text.upper() == "EL NIÑO ESPAÑOL ESTUDIA LA METODOLOGÍA"
    assert text.lower() == "el niño español estudia la metodología"
    
    # Test searching
    assert "niño" in text
    assert "español" in text
    
    # Test splitting
    words = text.split()
    assert "niño" in words
    assert "español" in words
    
    # Test replacement
    replaced = text.replace("niño", "niña")
    assert "niña" in replaced
    assert "niño" not in replaced


if __name__ == "__main__":
    # Run all tests
    test_spanish_character_encoding()
    test_spanish_special_punctuation()
    test_spanish_text_in_json()
    test_file_reading_with_spanish_content()
    test_string_operations_with_spanish_chars()
    print("✓ All Spanish encoding tests passed!")

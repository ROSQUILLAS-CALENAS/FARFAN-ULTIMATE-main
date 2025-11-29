# Spanish Character Encoding Support

## Overview

This project extensively uses Spanish language content, particularly for processing Colombian DNP (Departamento Nacional de Planeación) documents and the Decálogo de Derechos Humanos framework. Proper support for Spanish characters is critical for the system's functionality.

## Spanish Characters Used

The system handles all Spanish-specific characters, including:

- **Vowels with accents**: á, é, í, ó, ú (and their uppercase versions Á, É, Í, Ó, Ú)
- **Letter ñ**: Both lowercase (ñ) and uppercase (Ñ) - unique to Spanish
- **Special punctuation**: 
  - Opening question mark: ¿
  - Opening exclamation: ¡
- **Diaeresis**: ü (as in "bilingüe")

## Test Case: "ññ"

The string "ññ" (double ñ) serves as a comprehensive test case because:
1. It contains multiple instances of the most distinctively Spanish character
2. It tests proper UTF-8 encoding/decoding
3. It verifies that special characters don't cause issues in concatenation or repetition

## Implementation Details

### Encoding Standard

All Python files in this project use UTF-8 encoding (Python 3 default). This ensures:
- Proper storage and retrieval of Spanish text
- Correct JSON serialization with `ensure_ascii=False`
- Seamless file I/O operations

### Key Files with Spanish Content

1. **Cuestionario Original de la Metodología.md** - Contains evaluation questions in Spanish
2. **Decálogo de Derechos Humanos_ Puntos y Clústeres.md** - Human rights framework documentation
3. **decalogo_question_registry.py** - Question registry with Spanish metadata
4. **normative_validator.py** - Validation module with Spanish docstrings and comments
5. **EXTRACTOR DE EVIDENCIAS CONTEXTUAL.py** - Evidence extraction with Spanish text processing

### Testing

A comprehensive test suite is available in `tests/test_spanish_encoding.py` that verifies:
- Character encoding/decoding roundtrips
- JSON serialization
- File reading with Spanish content
- String operations (upper/lower case, search, split, replace)
- Special punctuation marks

Run the tests with:
```bash
python3 tests/test_spanish_encoding.py
```

Or with pytest (if available):
```bash
pytest tests/test_spanish_encoding.py -v
```

## Best Practices

1. **Always use UTF-8**: When opening files, explicitly specify encoding:
   ```python
   with open('file.txt', 'r', encoding='utf-8') as f:
       content = f.read()
   ```

2. **JSON with Spanish text**: Use `ensure_ascii=False`:
   ```python
   json.dumps(data, ensure_ascii=False)
   ```

3. **Database storage**: Ensure database charset is UTF-8 (utf8mb4 for MySQL)

4. **API responses**: Set proper Content-Type headers:
   ```
   Content-Type: application/json; charset=utf-8
   ```

## Troubleshooting

If you encounter encoding issues:

1. **Verify system locale**:
   ```bash
   python3 -c "import sys; print(sys.getdefaultencoding())"
   # Should output: utf-8
   ```

2. **Check file encoding**:
   ```bash
   file -i filename.py
   # Should show: charset=utf-8
   ```

3. **Test Spanish character support**:
   ```bash
   python3 -c "print('ññ')"
   # Should output: ññ
   ```

## Related Documentation

- [README_METHODOLOGY.md](../README_METHODOLOGY.md) - DNP methodology documentation
- [Cuestionario Original de la Metodología.md](../Cuestionario%20Original%20de%20la%20Metodología.md)
- [Decálogo de Derechos Humanos_ Puntos y Clústeres.md](../Decálogo%20de%20Derechos%20Humanos_%20Puntos%20y%20Clústeres.md)

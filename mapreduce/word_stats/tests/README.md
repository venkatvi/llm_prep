# MapReduce Tests

This directory contains comprehensive unit tests for the MapReduce word count implementation.

## 📁 Test Structure

```
tests/
├── __init__.py              # Package initialization
├── test_word_count.py       # Core MapReduce function tests
└── README.md               # This file
```

## 🧪 Test Coverage

### Core MapReduce Functions
- **`test_map_word_count()`** - Tests map phase word extraction
- **`test_reduce_word_count()`** - Tests reduce phase aggregation
- **`test_reduce_across_files()`** - Tests multi-file aggregation

### Edge Cases
- **`test_empty_input()`** - Handles empty lines and whitespace
- **`test_single_word_line()`** - Single word processing
- **`test_multiple_spaces()`** - Multiple spaces between words

## 🚀 Running Tests

### Method 1: Standalone Test Runner
```bash
python run_tests.py
```

### Method 2: Direct Test Module
```bash
python tests/test_word_count.py
```

### Method 3: From Main Application
```bash
python word_count.py  # Tests run automatically before processing
```

### Method 4: Using pytest (if installed)
```bash
pip install pytest
pytest tests/
```

## 📊 Expected Output

```
==================================================
RUNNING MAPREDUCE UNIT TESTS
==================================================
✓ test_map_word_count passed
✓ test_reduce_word_count passed
✓ test_reduce_across_files passed
✓ test_empty_input passed
✓ test_single_word_line passed
✓ test_multiple_spaces passed

Test Results: 6/6 tests passed
🎉 All tests passed!
==================================================
```

## ➕ Adding New Tests

To add new tests:

1. Create test functions in `test_word_count.py`
2. Follow naming convention: `test_function_name()`
3. Add to the `test_functions` list in `run_all_tests()`
4. Use descriptive docstrings and assertions

### Example Test Template
```python
def test_new_feature():
    """
    Test description here.

    Explains what the test validates.
    """
    # Arrange
    input_data = "test input"
    expected = [("expected", "result")]

    # Act
    result = function_to_test(input_data)

    # Assert
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ test_new_feature passed")
```

## 🔧 Test Guidelines

1. **Isolation**: Each test should be independent
2. **Clarity**: Use descriptive names and docstrings
3. **Coverage**: Test normal cases, edge cases, and error conditions
4. **Assertions**: Include clear error messages
5. **Feedback**: Print success messages for passed tests

## 📈 Future Test Enhancements

Potential additions for Level 2 and beyond:
- Memory usage tests
- Performance benchmarks
- Large file processing tests
- Error handling tests
- Parallel processing validation
- Custom partitioner tests
- Shuffle phase verification
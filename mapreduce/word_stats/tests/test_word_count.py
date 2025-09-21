"""
Unit tests for MapReduce framework implementation.

This module contains comprehensive tests for all core MapReduce functions
including map phase, reduce phase, multi-file aggregation, and the new
class-based architecture.
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from factories.registry import get_mapreduce_class, reduce_across_files
from factories.word_count import WordCountMapReduce


def test_map_word_count():
    """
    Test the map phase with known input.

    Verifies that the WordCountMapReduce.map method correctly splits text
    and emits (word, 1) pairs for each word.
    """
    result = list(WordCountMapReduce.map("hello world hello"))
    expected = [("hello", 1), ("world", 1), ("hello", 1)]
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ test_map_word_count passed")


def test_reduce_word_count():
    """
    Test the reduce phase aggregation.

    Verifies that the WordCountMapReduce.reduce method correctly aggregates
    word counts from multiple generators.
    """
    gen1 = WordCountMapReduce.map("hello world")
    gen2 = WordCountMapReduce.map("hello test")
    result = WordCountMapReduce.reduce([gen1, gen2])

    assert result["hello"] == 2, f"Expected hello=2, got {result['hello']}"
    assert result["world"] == 1, f"Expected world=1, got {result['world']}"
    assert result["test"] == 1, f"Expected test=1, got {result['test']}"
    print("✓ test_reduce_word_count passed")


def test_reduce_across_files():
    """
    Test the final aggregation across multiple files.

    Verifies that reduce_across_files correctly combines word counts
    from multiple file processing results.
    """
    file1_counts = {"hello": 2, "world": 1}
    file2_counts = {"hello": 1, "test": 3}
    result = reduce_across_files([file1_counts, file2_counts], "word_count")

    expected = {"hello": 3, "world": 1, "test": 3}
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ test_reduce_across_files passed")


def test_empty_input():
    """
    Test edge cases with empty input.

    Verifies that the system handles empty lines and files gracefully.
    """
    # Test empty line
    result = list(WordCountMapReduce.map(""))
    assert result == [], f"Expected empty list for empty input, got {result}"

    # Test whitespace-only line
    result = list(WordCountMapReduce.map("   \n\t  "))
    assert result == [], f"Expected empty list for whitespace input, got {result}"

    print("✓ test_empty_input passed")


def test_single_word_line():
    """
    Test processing of single word lines.

    Ensures that lines with only one word are processed correctly.
    """
    result = list(WordCountMapReduce.map("hello"))
    expected = [("hello", 1)]
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ test_single_word_line passed")


def test_multiple_spaces():
    """
    Test handling of multiple spaces between words.

    Verifies that multiple spaces are handled correctly by split().
    """
    result = list(WordCountMapReduce.map("hello    world     test"))
    expected = [("hello", 1), ("world", 1), ("test", 1)]
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ test_multiple_spaces passed")


def run_all_tests():
    """
    Run all test functions and report results.

    Executes all defined test functions and provides a summary
    of test results.

    Returns:
        bool: True if all tests passed, False otherwise
    """
    print("\n" + "=" * 50)
    print("RUNNING MAPREDUCE UNIT TESTS")
    print("=" * 50)

    test_functions = [
        test_map_word_count,
        test_reduce_word_count,
        test_reduce_across_files,
        test_empty_input,
        test_single_word_line,
        test_multiple_spaces,
    ]

    passed = 0
    total = len(test_functions)

    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed: {e}")

    print(f"\nTest Results: {passed}/{total} tests passed")
    if passed == total:
        print("🎉 All tests passed!")
        success = True
    else:
        print(f"⚠ {total - passed} tests failed")
        success = False

    print("=" * 50)
    return success


if __name__ == "__main__":
    """
    Run tests when module is executed directly.
    """
    success = run_all_tests()
    sys.exit(0 if success else 1)

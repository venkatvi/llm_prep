"""
Performance and edge case tests for MapReduce framework.

This module tests performance characteristics, edge cases, and error handling
for the MapReduce implementation.
"""

import sys
import time
import tempfile
import os
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from factories.registry import get_mapreduce_class, reduce_across_files
from factories.word_count import WordCountMapReduce
from map_reduce_framework import get_words_stats_in_file


def test_large_input_performance():
    """Test performance with large input data."""
    print("Testing large input performance...")

    # Generate large text
    large_text = " ".join(["word" + str(i) for i in range(1000)])

    # Time the map operation
    start_time = time.time()
    result = list(WordCountMapReduce.map(large_text))
    map_time = time.time() - start_time

    # Verify correctness
    assert len(result) == 1000, f"Expected 1000 words, got {len(result)}"

    # Time should be reasonable (less than 1 second for 1000 words)
    assert map_time < 1.0, f"Map operation too slow: {map_time:.3f}s for 1000 words"

    print(f"✓ Large input performance test passed (map: {map_time:.3f}s)")


def test_memory_efficiency():
    """Test memory efficiency with generators."""
    print("Testing memory efficiency...")

    # Test that map returns a generator, not a list
    result = WordCountMapReduce.map("hello world")
    assert hasattr(result, '__iter__') and hasattr(result, '__next__'), "Map should return generator"

    # Test that generators can be consumed multiple times by converting to list
    gen1 = WordCountMapReduce.map("hello world")
    gen2 = WordCountMapReduce.map("hello world")

    list1 = list(gen1)
    list2 = list(gen2)
    assert list1 == list2, "Generators should produce same results"

    print("✓ Memory efficiency tests passed")


def test_unicode_and_special_characters():
    """Test handling of unicode and special characters."""
    print("Testing unicode and special characters...")

    # Test unicode characters
    result = list(WordCountMapReduce.map("héllo wörld"))
    expected = [("héllo", 1), ("wörld", 1)]
    assert result == expected, f"Unicode failed: expected {expected}, got {result}"

    # Test with numbers and punctuation (split() handles these as separate words)
    result = list(WordCountMapReduce.map("hello123 world! test?"))
    expected = [("hello123", 1), ("world!", 1), ("test?", 1)]
    assert result == expected, f"Special chars failed: expected {expected}, got {result}"

    # Test emoji
    result = list(WordCountMapReduce.map("hello 🌍 world"))
    expected = [("hello", 1), ("🌍", 1), ("world", 1)]
    assert result == expected, f"Emoji failed: expected {expected}, got {result}"

    print("✓ Unicode and special character tests passed")


def test_extreme_edge_cases():
    """Test extreme edge cases."""
    print("Testing extreme edge cases...")

    # Test very long word
    long_word = "a" * 1000
    result = list(WordCountMapReduce.map(long_word))
    expected = [(long_word, 1)]
    assert result == expected, "Long word test failed"

    # Test line with only spaces and tabs
    result = list(WordCountMapReduce.map("\t   \t  "))
    assert result == [], "Whitespace-only line should return empty list"

    # Test newlines and carriage returns
    result = list(WordCountMapReduce.map("hello\nworld\r\ntest"))
    expected = [("hello", 1), ("world", 1), ("test", 1)]
    assert result == expected, f"Newline handling failed: expected {expected}, got {result}"

    # Test empty reduce input
    result = WordCountMapReduce.reduce([])
    assert result == {}, "Empty reduce should return empty dict"

    # Test empty reduce_all input
    result = WordCountMapReduce.reduce_all([])
    assert result == {}, "Empty reduce_all should return empty dict"

    print("✓ Extreme edge case tests passed")


def test_error_handling():
    """Test error handling and invalid inputs."""
    print("Testing error handling...")

    # Test invalid stats_type
    try:
        get_mapreduce_class("invalid_stats_type")
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError:
        pass  # Expected

    # Test None input (should raise error)
    try:
        list(WordCountMapReduce.map(None))
        assert False, "Should have raised an error for None input"
    except (TypeError, AttributeError):
        pass  # Expected

    print("✓ Error handling tests passed")


def test_consistency_across_operations():
    """Test consistency between different operation modes."""
    print("Testing consistency across operations...")

    text_lines = ["hello world", "hello test", "world"]

    # Method 1: Direct class usage
    mapreduce_class = get_mapreduce_class("word_count")
    generators = [mapreduce_class.map(line) for line in text_lines]
    per_line_results = [mapreduce_class.reduce([gen]) for gen in generators]
    result1 = reduce_across_files(per_line_results, "word_count")

    # Method 2: Using functools.reduce
    mapreduce_class = get_mapreduce_class("word_count")
    generators = [mapreduce_class.map(line) for line in text_lines]
    per_line_results = [mapreduce_class.reduce([gen], use_reduce=True) for gen in generators]
    result2 = reduce_across_files(per_line_results, "word_count", use_reduce=True)

    assert result1 == result2, f"Inconsistent results: {result1} vs {result2}"

    print("✓ Consistency tests passed")


def test_file_processing_edge_cases():
    """Test file processing edge cases."""
    print("Testing file processing edge cases...")

    # Test with temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("hello world\ntest line")
        temp_file = f.name

    try:
        # Test file processing
        result = get_words_stats_in_file([temp_file], "word_count")
        expected = {"hello": 1, "world": 1, "test": 1, "line": 1}
        assert result == expected, f"File processing failed: expected {expected}, got {result}"

        # Test with empty file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            empty_file = f.name

        try:
            result = get_words_stats_in_file([empty_file], "word_count")
            assert result == {}, f"Empty file should return empty dict, got {result}"
        finally:
            os.unlink(empty_file)

    finally:
        os.unlink(temp_file)

    print("✓ File processing edge case tests passed")


def test_numerical_precision():
    """Test numerical precision for floating point calculations."""
    print("Testing numerical precision...")

    # Test average calculation precision
    mapreduce_class = get_mapreduce_class("average_word_length")

    # Create data that would cause precision issues
    file_stats = [(1, 3), (2, 3), (3, 3)]  # Should give average of 2.0
    result = reduce_across_files(file_stats, "average_word_length")
    expected = 6.0 / 9.0  # 0.6666...

    assert abs(result - expected) < 1e-10, f"Precision error: expected {expected}, got {result}"

    # Test zero division
    result = mapreduce_class.reduce_all([(0, 0)])
    assert result == 0.0, f"Zero division should return 0.0, got {result}"

    print("✓ Numerical precision tests passed")


def test_performance_comparison():
    """Test performance comparison between different approaches."""
    print("Testing performance comparison...")

    # Generate test data
    test_lines = ["hello world test"] * 100

    # Time normal reduce
    start_time = time.time()
    for line in test_lines:
        gen = WordCountMapReduce.map(line)
        WordCountMapReduce.reduce([gen], use_reduce=False)
    normal_time = time.time() - start_time

    # Time functional reduce
    start_time = time.time()
    for line in test_lines:
        gen = WordCountMapReduce.map(line)
        WordCountMapReduce.reduce([gen], use_reduce=True)
    functional_time = time.time() - start_time

    # Both should complete in reasonable time
    assert normal_time < 1.0, f"Normal reduce too slow: {normal_time:.3f}s"
    assert functional_time < 1.0, f"Functional reduce too slow: {functional_time:.3f}s"

    print(f"✓ Performance comparison passed (normal: {normal_time:.3f}s, functional: {functional_time:.3f}s)")


def run_all_tests():
    """Run all performance and edge case tests."""
    print("\n" + "=" * 60)
    print("RUNNING PERFORMANCE & EDGE CASE TESTS")
    print("=" * 60)

    test_functions = [
        test_large_input_performance,
        test_memory_efficiency,
        test_unicode_and_special_characters,
        test_extreme_edge_cases,
        test_error_handling,
        test_consistency_across_operations,
        test_file_processing_edge_cases,
        test_numerical_precision,
        test_performance_comparison,
    ]

    passed = 0
    total = len(test_functions)

    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nTest Results: {passed}/{total} tests passed")
    if passed == total:
        print("🎉 All performance and edge case tests passed!")
        success = True
    else:
        print(f"⚠ {total - passed} tests failed")
        success = False

    print("=" * 60)
    return success


if __name__ == "__main__":
    """Run tests when module is executed directly."""
    success = run_all_tests()
    sys.exit(0 if success else 1)
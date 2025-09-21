"""
Tests for the factory pattern and registry system.

This module tests the factory pattern implementation, registry functions,
and integration between different components of the system.
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from factories.registry import (
    get_mapreduce_class,
    reduce_across_files,
    reduce_shuffled_word_stats,
    shuffle_results,
)
from factories.word_count import WordCountMapReduce
from factories.word_length_sum import WordLengthSumMapReduce
from factories.word_length_average import WordLengthAverageMapReduce


def test_get_mapreduce_class():
    """Test the factory function returns correct classes."""
    print("Testing get_mapreduce_class factory...")

    # Test word_count
    cls = get_mapreduce_class("word_count")
    assert cls == WordCountMapReduce, f"Expected WordCountMapReduce, got {cls}"

    # Test sum_of_word_lengths
    cls = get_mapreduce_class("sum_of_word_lengths")
    assert cls == WordLengthSumMapReduce, f"Expected WordLengthSumMapReduce, got {cls}"

    # Test average_word_length
    cls = get_mapreduce_class("average_word_length")
    assert cls == WordLengthAverageMapReduce, f"Expected WordLengthAverageMapReduce, got {cls}"

    # Test invalid stats_type
    try:
        get_mapreduce_class("invalid_type")
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError:
        pass  # Expected

    print("✓ get_mapreduce_class tests passed")


def test_factory_method_consistency():
    """Test that factory-retrieved classes work the same as direct imports."""
    print("Testing factory method consistency...")

    # Test word_count consistency
    direct_class = WordCountMapReduce
    factory_class = get_mapreduce_class("word_count")

    # Test map method
    direct_result = list(direct_class.map("hello world"))
    factory_result = list(factory_class.map("hello world"))
    assert direct_result == factory_result, "Map methods inconsistent"

    # Test reduce method
    gen1 = direct_class.map("hello")
    gen2 = factory_class.map("world")
    direct_result = direct_class.reduce([gen1])
    factory_result = factory_class.reduce([gen2])
    assert type(direct_result) == type(factory_result), "Reduce methods inconsistent"

    print("✓ Factory method consistency tests passed")


def test_shuffle_results():
    """Test the shuffle_results function."""
    print("Testing shuffle_results...")

    # Create generators
    gen1 = WordCountMapReduce.map("hello world")
    gen2 = WordCountMapReduce.map("hello test")

    result = shuffle_results([gen1, gen2])

    expected = {
        "hello": [1, 1],
        "world": [1],
        "test": [1]
    }
    assert result == expected, f"Expected {expected}, got {result}"

    print("✓ shuffle_results tests passed")


def test_reduce_shuffled_word_stats():
    """Test the reduce_shuffled_word_stats function."""
    print("Testing reduce_shuffled_word_stats...")

    shuffled_data = {"hello": [1, 1, 1], "world": [1], "test": [1, 1]}

    # Test word_count
    result = reduce_shuffled_word_stats(shuffled_data, False, "word_count")
    expected = {"hello": 3, "world": 1, "test": 2}
    assert result == expected, f"Word count failed: expected {expected}, got {result}"

    # Test sum_of_word_lengths
    shuffled_data = {"hello": [5, 5], "world": [5], "test": [4]}
    result = reduce_shuffled_word_stats(shuffled_data, False, "sum_of_word_lengths")
    expected = (19, 4)  # total_chars=19, total_words=4
    assert result == expected, f"Sum failed: expected {expected}, got {result}"

    # Test average_word_length
    result = reduce_shuffled_word_stats(shuffled_data, False, "average_word_length")
    expected = (19, 4)  # returns totals, not average
    assert result == expected, f"Average failed: expected {expected}, got {result}"

    print("✓ reduce_shuffled_word_stats tests passed")


def test_reduce_across_files():
    """Test the reduce_across_files function with different stats types."""
    print("Testing reduce_across_files...")

    # Test word_count
    file1_counts = {"hello": 2, "world": 1}
    file2_counts = {"hello": 1, "test": 3}
    result = reduce_across_files([file1_counts, file2_counts], "word_count")
    expected = {"hello": 3, "world": 1, "test": 3}
    assert result == expected, f"Word count failed: expected {expected}, got {result}"

    # Test sum_of_word_lengths
    file1_stats = (10, 2)
    file2_stats = (8, 1)
    result = reduce_across_files([file1_stats, file2_stats], "sum_of_word_lengths")
    expected = (18, 3)
    assert result == expected, f"Sum failed: expected {expected}, got {result}"

    # Test average_word_length
    result = reduce_across_files([file1_stats, file2_stats], "average_word_length")
    expected = 18.0 / 3.0  # 6.0
    assert abs(result - expected) < 0.001, f"Average failed: expected {expected}, got {result}"

    print("✓ reduce_across_files tests passed")


def test_end_to_end_pipeline():
    """Test a complete end-to-end pipeline for each stats type."""
    print("Testing end-to-end pipeline...")

    text_lines = ["hello world", "hello test", "world test"]

    # Test word_count pipeline
    mapreduce_class = get_mapreduce_class("word_count")
    generators = [mapreduce_class.map(line) for line in text_lines]
    per_line_results = [mapreduce_class.reduce([gen]) for gen in generators]
    final_result = reduce_across_files(per_line_results, "word_count")
    expected = {"hello": 2, "world": 2, "test": 2}
    assert final_result == expected, f"Word count pipeline failed: expected {expected}, got {final_result}"

    # Test sum_of_word_lengths pipeline
    mapreduce_class = get_mapreduce_class("sum_of_word_lengths")
    generators = [mapreduce_class.map(line) for line in text_lines]
    per_line_results = [mapreduce_class.reduce([gen]) for gen in generators]
    final_result = reduce_across_files(per_line_results, "sum_of_word_lengths")
    expected = (28, 6)  # hello(5) + world(5) + hello(5) + test(4) + world(5) + test(4) = 28 chars, 6 words
    assert final_result == expected, f"Sum pipeline failed: expected {expected}, got {final_result}"

    # Test average_word_length pipeline
    mapreduce_class = get_mapreduce_class("average_word_length")
    generators = [mapreduce_class.map(line) for line in text_lines]
    per_line_results = [mapreduce_class.reduce([gen]) for gen in generators]
    final_result = reduce_across_files(per_line_results, "average_word_length")
    expected = 28.0 / 6.0  # ~4.67
    assert abs(final_result - expected) < 0.001, f"Average pipeline failed: expected {expected}, got {final_result}"

    print("✓ End-to-end pipeline tests passed")


def test_shuffle_pipeline():
    """Test pipeline with shuffle phase."""
    print("Testing shuffle pipeline...")

    text_lines = ["hello world", "hello test"]

    # Get generators
    mapreduce_class = get_mapreduce_class("word_count")
    generators = [mapreduce_class.map(line) for line in text_lines]

    # Shuffle
    shuffled = shuffle_results(generators)
    expected_shuffled = {"hello": [1, 1], "world": [1], "test": [1]}
    assert shuffled == expected_shuffled, f"Shuffle failed: expected {expected_shuffled}, got {shuffled}"

    # Reduce shuffled
    result = reduce_shuffled_word_stats(shuffled, False, "word_count")
    expected = {"hello": 2, "world": 1, "test": 1}
    assert result == expected, f"Shuffle reduce failed: expected {expected}, got {result}"

    print("✓ Shuffle pipeline tests passed")


def run_all_tests():
    """Run all factory pattern and integration tests."""
    print("\n" + "=" * 60)
    print("RUNNING FACTORY PATTERN & INTEGRATION TESTS")
    print("=" * 60)

    test_functions = [
        test_get_mapreduce_class,
        test_factory_method_consistency,
        test_shuffle_results,
        test_reduce_shuffled_word_stats,
        test_reduce_across_files,
        test_end_to_end_pipeline,
        test_shuffle_pipeline,
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
        print("🎉 All factory pattern tests passed!")
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
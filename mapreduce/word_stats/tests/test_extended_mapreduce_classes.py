"""
Tests for extended MapReduce class implementations.

This module tests the additional MapReduce classes that extend the basic
functionality: WordTopKMapReduce and FrequencyCountMapReduce.
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from factories.registry import get_mapreduce_class
from factories.word_count import WordCountMapReduce, WordTopKMapReduce, FrequencyCountMapReduce


def test_word_topk_mapreduce():
    """Test WordTopKMapReduce class functionality."""
    print("Testing WordTopKMapReduce...")

    # Test that it inherits map functionality from WordCountMapReduce
    result = list(WordTopKMapReduce.map("hello world hello test world"))
    expected = [("hello", 1), ("world", 1), ("hello", 1), ("test", 1), ("world", 1)]
    assert result == expected, f"Map failed: expected {expected}, got {result}"

    # Test reduce functionality (inherited)
    gen1 = WordTopKMapReduce.map("hello world hello")
    gen2 = WordTopKMapReduce.map("test world test test")
    result = WordTopKMapReduce.reduce([gen1, gen2])
    expected = {"hello": 2, "world": 2, "test": 3}
    assert result == expected, f"Reduce failed: expected {expected}, got {result}"

    # Test reduce_all with top_k functionality
    file1_counts = {"hello": 5, "world": 3, "test": 2, "foo": 1}
    file2_counts = {"hello": 2, "world": 1, "bar": 4, "baz": 1}

    # Test top 3
    result = WordTopKMapReduce.reduce_all([file1_counts, file2_counts], use_reduce=False, top_k=3)
    # Expected: hello=7, bar=4, world=4 (top 3)
    assert len(result) == 3, f"Expected 3 items, got {len(result)}"
    assert result["hello"] == 7, f"Expected hello=7, got {result.get('hello')}"
    assert "bar" in result, "Expected 'bar' in top 3"
    assert "world" in result, "Expected 'world' in top 3"

    # Test top 2
    result = WordTopKMapReduce.reduce_all([file1_counts, file2_counts], use_reduce=False, top_k=2)
    assert len(result) == 2, f"Expected 2 items, got {len(result)}"
    assert result["hello"] == 7, f"Expected hello=7, got {result.get('hello')}"

    # Test that results are sorted by frequency (descending)
    result = WordTopKMapReduce.reduce_all([file1_counts, file2_counts], use_reduce=False, top_k=5)
    values = list(result.values())
    assert values == sorted(values, reverse=True), f"Results not sorted by frequency: {values}"

    print("✓ WordTopKMapReduce tests passed")


def test_frequency_count_mapreduce():
    """Test FrequencyCountMapReduce class functionality."""
    print("Testing FrequencyCountMapReduce...")

    # Test that it inherits map functionality from WordCountMapReduce
    result = list(FrequencyCountMapReduce.map("hello world hello test"))
    expected = [("hello", 1), ("world", 1), ("hello", 1), ("test", 1)]
    assert result == expected, f"Map failed: expected {expected}, got {result}"

    # Test reduce functionality (inherited)
    gen1 = FrequencyCountMapReduce.map("hello world hello")
    gen2 = FrequencyCountMapReduce.map("test world test test")
    result = FrequencyCountMapReduce.reduce([gen1, gen2])
    expected = {"hello": 2, "world": 2, "test": 3}
    assert result == expected, f"Reduce failed: expected {expected}, got {result}"

    # Test reduce_all with frequency counting functionality
    file1_counts = {"hello": 3, "world": 2, "test": 1, "foo": 1}  # freq 3:1, freq 2:1, freq 1:2
    file2_counts = {"hello": 1, "bar": 2, "baz": 1, "qux": 1}     # freq 1:3, freq 2:1

    # Combined: hello=4, world=2, bar=2, test=1, foo=1, baz=1, qux=1
    # Frequencies: freq 4:1, freq 2:2, freq 1:4
    result = FrequencyCountMapReduce.reduce_all([file1_counts, file2_counts])
    expected = {1: 4, 2: 2, 4: 1}  # 4 words appear 1 time, 2 words appear 2 times, 1 word appears 4 times
    assert result == expected, f"Frequency count failed: expected {expected}, got {result}"

    # Test that results are sorted by frequency value
    result_keys = list(result.keys())
    assert result_keys == sorted(result_keys), f"Results not sorted by frequency: {result_keys}"

    # Test edge case: empty input
    result = FrequencyCountMapReduce.reduce_all([])
    assert result == {}, f"Empty input should return empty dict, got {result}"

    # Test single frequency case
    single_freq_data = [{"a": 5, "b": 5, "c": 5}]
    result = FrequencyCountMapReduce.reduce_all(single_freq_data)
    expected = {5: 3}  # 3 words all appear 5 times
    assert result == expected, f"Single frequency failed: expected {expected}, got {result}"

    print("✓ FrequencyCountMapReduce tests passed")


def test_factory_integration_extended_classes():
    """Test factory integration with extended MapReduce classes."""
    print("Testing factory integration with extended classes...")

    # Test topk factory
    topk_class = get_mapreduce_class("topk")
    assert topk_class == WordTopKMapReduce, f"Expected WordTopKMapReduce, got {topk_class}"

    # Test freq_count factory
    freq_class = get_mapreduce_class("freq_count")
    assert freq_class == FrequencyCountMapReduce, f"Expected FrequencyCountMapReduce, got {freq_class}"

    # Test that factory classes work the same as direct imports
    direct_result = list(WordTopKMapReduce.map("hello world"))
    factory_result = list(topk_class.map("hello world"))
    assert direct_result == factory_result, "Factory and direct import give different results"

    direct_result = list(FrequencyCountMapReduce.map("hello world"))
    factory_result = list(freq_class.map("hello world"))
    assert direct_result == factory_result, "Factory and direct import give different results"

    print("✓ Factory integration tests passed")


def test_local_combiner_methods():
    """Test the get_modified_stats_type_for_local_combiner methods."""
    print("Testing local combiner methods...")

    # Test WordTopKMapReduce
    result = WordTopKMapReduce.get_modified_stats_type_for_local_combiner()
    assert result == "word_count", f"Expected 'word_count', got {result}"

    # Test FrequencyCountMapReduce
    result = FrequencyCountMapReduce.get_modified_stats_type_for_local_combiner()
    assert result == "word_count", f"Expected 'word_count', got {result}"

    print("✓ Local combiner method tests passed")


def test_extended_classes_edge_cases():
    """Test edge cases for extended MapReduce classes."""
    print("Testing edge cases for extended classes...")

    # Test WordTopKMapReduce with k larger than number of unique words
    file_counts = {"hello": 3, "world": 2}
    result = WordTopKMapReduce.reduce_all([file_counts], top_k=10)
    assert len(result) == 2, f"Expected 2 items (all available), got {len(result)}"
    assert result == {"hello": 3, "world": 2}, f"Unexpected result: {result}"

    # Test WordTopKMapReduce with k=0
    result = WordTopKMapReduce.reduce_all([file_counts], top_k=0)
    assert result == {}, f"Expected empty dict for k=0, got {result}"

    # Test FrequencyCountMapReduce with single word
    single_word_data = [{"hello": 5}]
    result = FrequencyCountMapReduce.reduce_all(single_word_data)
    expected = {5: 1}  # 1 word appears 5 times
    assert result == expected, f"Single word failed: expected {expected}, got {result}"

    # Test FrequencyCountMapReduce with same frequencies
    same_freq_data = [{"a": 2, "b": 2, "c": 2}]
    result = FrequencyCountMapReduce.reduce_all(same_freq_data)
    expected = {2: 3}  # 3 words all appear 2 times
    assert result == expected, f"Same frequencies failed: expected {expected}, got {result}"

    print("✓ Extended classes edge case tests passed")


def test_inheritance_behavior():
    """Test that inheritance works correctly."""
    print("Testing inheritance behavior...")

    # Test that WordTopKMapReduce inherits from WordCountMapReduce
    assert issubclass(WordTopKMapReduce, WordCountMapReduce), "WordTopKMapReduce should inherit from WordCountMapReduce"

    # Test that FrequencyCountMapReduce inherits from WordCountMapReduce
    assert issubclass(FrequencyCountMapReduce, WordCountMapReduce), "FrequencyCountMapReduce should inherit from WordCountMapReduce"

    # Test that inherited methods work
    topk_instance = WordTopKMapReduce()
    freq_instance = FrequencyCountMapReduce()

    # Test map method (inherited)
    result1 = list(topk_instance.map("hello world"))
    result2 = list(freq_instance.map("hello world"))
    expected = [("hello", 1), ("world", 1)]
    assert result1 == expected, "TopK map inheritance failed"
    assert result2 == expected, "FreqCount map inheritance failed"

    # Test reduce method (inherited)
    gen1 = topk_instance.map("hello")
    gen2 = freq_instance.map("world")
    result1 = topk_instance.reduce([gen1])
    result2 = freq_instance.reduce([gen2])
    assert isinstance(result1, dict), "TopK reduce inheritance failed"
    assert isinstance(result2, dict), "FreqCount reduce inheritance failed"

    print("✓ Inheritance behavior tests passed")


def run_all_tests():
    """Run all extended MapReduce class tests."""
    print("\n" + "=" * 60)
    print("RUNNING EXTENDED MAPREDUCE CLASS TESTS")
    print("=" * 60)

    test_functions = [
        test_word_topk_mapreduce,
        test_frequency_count_mapreduce,
        test_factory_integration_extended_classes,
        test_local_combiner_methods,
        test_extended_classes_edge_cases,
        test_inheritance_behavior,
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
        print("🎉 All extended MapReduce class tests passed!")
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
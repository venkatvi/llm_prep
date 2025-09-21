"""
Comprehensive tests for MapReduce class implementations.

This module tests all MapReduce classes (WordCountMapReduce, WordLengthSumMapReduce,
WordLengthAverageMapReduce) and their static methods.
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from factories.word_count import WordCountMapReduce
from factories.word_length_sum import WordLengthSumMapReduce
from factories.word_length_average import WordLengthAverageMapReduce


def test_word_count_mapreduce():
    """Test all methods of WordCountMapReduce class."""
    print("Testing WordCountMapReduce...")

    # Test map method
    result = list(WordCountMapReduce.map("hello world hello"))
    expected = [("hello", 1), ("world", 1), ("hello", 1)]
    assert result == expected, f"Map failed: expected {expected}, got {result}"

    # Test reduce method
    gen1 = WordCountMapReduce.map("hello world")
    gen2 = WordCountMapReduce.map("hello test")
    result = WordCountMapReduce.reduce([gen1, gen2])
    assert result["hello"] == 2, f"Reduce failed: expected hello=2, got {result['hello']}"
    assert result["world"] == 1, f"Reduce failed: expected world=1, got {result['world']}"
    assert result["test"] == 1, f"Reduce failed: expected test=1, got {result['test']}"

    # Test reduce_all method
    file1_counts = {"hello": 2, "world": 1}
    file2_counts = {"hello": 1, "test": 3}
    result = WordCountMapReduce.reduce_all([file1_counts, file2_counts])
    expected = {"hello": 3, "world": 1, "test": 3}
    assert result == expected, f"ReduceAll failed: expected {expected}, got {result}"

    # Test reduce_shuffled method
    shuffled_data = {"hello": [1, 1, 1], "world": [1], "test": [1, 1]}
    result = WordCountMapReduce.reduce_shuffled(shuffled_data)
    expected = {"hello": 3, "world": 1, "test": 2}
    assert result == expected, f"ReduceShuffled failed: expected {expected}, got {result}"

    print("✓ WordCountMapReduce tests passed")


def test_word_length_sum_mapreduce():
    """Test all methods of WordLengthSumMapReduce class."""
    print("Testing WordLengthSumMapReduce...")

    # Test map method
    result = list(WordLengthSumMapReduce.map("hello world"))
    expected = [("hello", 5), ("world", 5)]
    assert result == expected, f"Map failed: expected {expected}, got {result}"

    # Test reduce method
    gen1 = WordLengthSumMapReduce.map("hello world")  # 5 + 5 = 10 chars, 2 words
    gen2 = WordLengthSumMapReduce.map("test")  # 4 chars, 1 word
    result = WordLengthSumMapReduce.reduce([gen1, gen2])
    expected = (14, 3)  # total_chars=14, total_words=3
    assert result == expected, f"Reduce failed: expected {expected}, got {result}"

    # Test reduce_all method
    file1_stats = (10, 2)  # 10 chars, 2 words
    file2_stats = (8, 1)   # 8 chars, 1 word
    result = WordLengthSumMapReduce.reduce_all([file1_stats, file2_stats])
    expected = (18, 3)  # total_chars=18, total_words=3
    assert result == expected, f"ReduceAll failed: expected {expected}, got {result}"

    # Test reduce_shuffled method
    shuffled_data = {"hello": [5, 5], "world": [5], "test": [4]}
    result = WordLengthSumMapReduce.reduce_shuffled(shuffled_data)
    expected = (19, 4)  # total_chars=19, total_words=4
    assert result == expected, f"ReduceShuffled failed: expected {expected}, got {result}"

    print("✓ WordLengthSumMapReduce tests passed")


def test_word_length_average_mapreduce():
    """Test all methods of WordLengthAverageMapReduce class."""
    print("Testing WordLengthAverageMapReduce...")

    # Test map method (same as sum)
    result = list(WordLengthAverageMapReduce.map("hello world"))
    expected = [("hello", 5), ("world", 5)]
    assert result == expected, f"Map failed: expected {expected}, got {result}"

    # Test reduce method (returns totals, not average)
    gen1 = WordLengthAverageMapReduce.map("hello world")  # 5 + 5 = 10 chars, 2 words
    gen2 = WordLengthAverageMapReduce.map("test")  # 4 chars, 1 word
    result = WordLengthAverageMapReduce.reduce([gen1, gen2])
    expected = (14, 3)  # total_chars=14, total_words=3
    assert result == expected, f"Reduce failed: expected {expected}, got {result}"

    # Test reduce_all method (calculates average)
    file1_stats = (10, 2)  # 10 chars, 2 words
    file2_stats = (8, 1)   # 8 chars, 1 word
    result = WordLengthAverageMapReduce.reduce_all([file1_stats, file2_stats])
    expected = 18.0 / 3.0  # average = 6.0
    assert abs(result - expected) < 0.001, f"ReduceAll failed: expected {expected}, got {result}"

    # Test reduce_shuffled method (returns totals, not average)
    shuffled_data = {"hello": [5, 5], "world": [5], "test": [4]}
    result = WordLengthAverageMapReduce.reduce_shuffled(shuffled_data)
    expected = (19, 4)  # total_chars=19, total_words=4
    assert result == expected, f"ReduceShuffled failed: expected {expected}, got {result}"

    print("✓ WordLengthAverageMapReduce tests passed")


def test_edge_cases():
    """Test edge cases for all MapReduce classes."""
    print("Testing edge cases...")

    # Test empty input for all classes
    for MapReduceClass in [WordCountMapReduce, WordLengthSumMapReduce, WordLengthAverageMapReduce]:
        result = list(MapReduceClass.map(""))
        assert result == [], f"{MapReduceClass.__name__} failed on empty input"

        result = list(MapReduceClass.map("   \t\n  "))
        assert result == [], f"{MapReduceClass.__name__} failed on whitespace input"

    # Test zero division handling for average
    result = WordLengthAverageMapReduce.reduce_all([])
    assert result == 0.0, f"Expected 0.0 for empty average, got {result}"

    result = WordLengthAverageMapReduce.reduce_all([(0, 0)])
    assert result == 0.0, f"Expected 0.0 for zero words, got {result}"

    print("✓ Edge case tests passed")


def test_functional_reduce_flag():
    """Test the use_reduce flag functionality."""
    print("Testing functional reduce flag...")

    # Test WordCountMapReduce with use_reduce=True
    gen1 = WordCountMapReduce.map("hello world")
    gen2 = WordCountMapReduce.map("hello test")
    result_normal = WordCountMapReduce.reduce([gen1, gen2], use_reduce=False)

    gen1 = WordCountMapReduce.map("hello world")
    gen2 = WordCountMapReduce.map("hello test")
    result_functional = WordCountMapReduce.reduce([gen1, gen2], use_reduce=True)

    assert result_normal == result_functional, f"Functional reduce gave different result: {result_normal} vs {result_functional}"

    # Test reduce_all with use_reduce=True
    file1_counts = {"hello": 2, "world": 1}
    file2_counts = {"hello": 1, "test": 3}
    result_normal = WordCountMapReduce.reduce_all([file1_counts, file2_counts], use_reduce=False)
    result_functional = WordCountMapReduce.reduce_all([file1_counts, file2_counts], use_reduce=True)

    assert result_normal == result_functional, f"Functional reduce_all gave different result"

    print("✓ Functional reduce flag tests passed")


def run_all_tests():
    """Run all MapReduce class tests."""
    print("\n" + "=" * 60)
    print("RUNNING MAPREDUCE CLASS TESTS")
    print("=" * 60)

    test_functions = [
        test_word_count_mapreduce,
        test_word_length_sum_mapreduce,
        test_word_length_average_mapreduce,
        test_edge_cases,
        test_functional_reduce_flag,
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
        print("🎉 All MapReduce class tests passed!")
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
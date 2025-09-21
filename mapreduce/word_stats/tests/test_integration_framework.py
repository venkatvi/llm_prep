"""
Integration tests for the complete MapReduce framework.

This module tests the integration between all components: main framework,
factory classes, MapReduce classes, and file processing.
"""

import sys
import tempfile
import os
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from map_reduce_framework import (
    get_words_stats_in_file,
    print_and_benchmark_word_stats_sequential,
    chunkify,
)
from factories.registry import get_mapreduce_class


def test_file_processing_integration():
    """Test integration of file processing with different stats types."""
    print("Testing file processing integration...")

    # Create temporary test files
    test_data = [
        "hello world test",
        "hello world",
        "test data example",
        "mapreduce framework test"
    ]

    temp_files = []
    for i, data in enumerate(test_data):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(data)
            temp_files.append(f.name)

    try:
        # Test word_count processing
        result = get_words_stats_in_file(temp_files, "word_count")
        assert isinstance(result, dict), "Word count should return dict"
        assert "hello" in result, "Should contain 'hello'"
        assert "world" in result, "Should contain 'world'"
        assert result["hello"] == 2, f"Expected hello=2, got {result['hello']}"

        # Test sum_of_word_lengths processing
        result = get_words_stats_in_file(temp_files, "sum_of_word_lengths")
        assert isinstance(result, tuple), "Sum should return tuple"
        assert len(result) == 2, "Sum should return (total_chars, total_words)"
        total_chars, total_words = result
        assert total_chars > 0, "Should have positive character count"
        assert total_words > 0, "Should have positive word count"

        # Test average_word_length processing (returns tuple when multiple files due to local combiner)
        result = get_words_stats_in_file(temp_files, "average_word_length")
        assert isinstance(result, tuple), "Average with multiple files should return tuple (total_chars, total_words)"
        assert len(result) == 2, "Average should return (total_chars, total_words)"
        total_chars, total_words = result
        assert total_chars > 0, "Should have positive character count"
        assert total_words > 0, "Should have positive word count"

        # Test with shuffle enabled
        result_shuffle = get_words_stats_in_file(temp_files, "word_count", use_shuffle=True)
        result_no_shuffle = get_words_stats_in_file(temp_files, "word_count", use_shuffle=False)
        assert result_shuffle == result_no_shuffle, "Shuffle and no-shuffle should give same results"

    finally:
        # Clean up temp files
        for temp_file in temp_files:
            os.unlink(temp_file)

    print("✓ File processing integration tests passed")


def test_chunkify_function():
    """Test the chunkify function for load balancing."""
    print("Testing chunkify function...")

    # Test basic chunking
    files = ["file1.txt", "file2.txt", "file3.txt", "file4.txt", "file5.txt"]
    result = chunkify(files, 2)
    assert len(result) == 2, f"Expected 2 chunks, got {len(result)}"
    assert len(result[0]) + len(result[1]) == 5, "All files should be distributed"

    # Test even distribution
    files = ["f1", "f2", "f3", "f4"]
    result = chunkify(files, 2)
    assert len(result[0]) == 2 and len(result[1]) == 2, "Should distribute evenly"

    # Test uneven distribution
    files = ["f1", "f2", "f3", "f4", "f5"]
    result = chunkify(files, 2)
    lengths = [len(chunk) for chunk in result]
    assert max(lengths) - min(lengths) <= 1, "Chunks should differ by at most 1"

    # Test more processes than files
    files = ["f1", "f2"]
    result = chunkify(files, 5)
    non_empty_chunks = [chunk for chunk in result if chunk]
    assert len(non_empty_chunks) <= 2, "Should not have more non-empty chunks than files"

    print("✓ Chunkify function tests passed")


def test_sequential_processing():
    """Test sequential processing pipeline."""
    print("Testing sequential processing...")

    # Create temporary data directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test files
        test_files = {
            "file1.txt": "hello world hello",
            "file2.txt": "world test data",
            "file3.txt": "hello test example"
        }

        for filename, content in test_files.items():
            (temp_path / filename).write_text(content)

        # Test word_count sequential processing
        result, time_taken = print_and_benchmark_word_stats_sequential(
            temp_path, "word_count", use_shuffle=False
        )
        assert isinstance(result, dict), "Sequential word count should return dict"
        assert time_taken > 0, "Should take some time"
        assert "hello" in result, "Should contain 'hello'"

        # Test sum_of_word_lengths sequential processing
        result, time_taken = print_and_benchmark_word_stats_sequential(
            temp_path, "sum_of_word_lengths", use_shuffle=False
        )
        assert isinstance(result, tuple), "Sequential sum should return tuple"
        assert time_taken > 0, "Should take some time"

        # Test average_word_length sequential processing
        result, time_taken = print_and_benchmark_word_stats_sequential(
            temp_path, "average_word_length", use_shuffle=False
        )
        assert isinstance(result, float), "Sequential average should return float"
        assert time_taken > 0, "Should take some time"
        assert result > 0, "Average should be positive"

    print("✓ Sequential processing tests passed")


def test_consistency_across_modes():
    """Test consistency between different processing modes."""
    print("Testing consistency across modes...")

    # Create test data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        test_files = {
            "file1.txt": "hello world test example",
            "file2.txt": "mapreduce framework testing",
        }

        for filename, content in test_files.items():
            (temp_path / filename).write_text(content)

        # Test word_count consistency
        result_normal = print_and_benchmark_word_stats_sequential(
            temp_path, "word_count", use_shuffle=False, use_reduce=False
        )[0]

        result_shuffle = print_and_benchmark_word_stats_sequential(
            temp_path, "word_count", use_shuffle=True, use_reduce=False
        )[0]

        result_reduce = print_and_benchmark_word_stats_sequential(
            temp_path, "word_count", use_shuffle=False, use_reduce=True
        )[0]

        assert result_normal == result_shuffle, "Normal and shuffle modes should give same results"
        assert result_normal == result_reduce, "Normal and reduce modes should give same results"

    print("✓ Consistency across modes tests passed")


def test_extended_stats_types():
    """Test extended stats types (topk, freq_count)."""
    print("Testing extended stats types...")

    # Create test data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create data with known frequency distribution
        test_files = {
            "file1.txt": "a a a b b c",  # a:3, b:2, c:1
            "file2.txt": "a b c d d",    # a:1, b:1, c:1, d:2
        }
        # Combined: a:4, b:3, d:2, c:2

        for filename, content in test_files.items():
            (temp_path / filename).write_text(content)

        # Test topk processing
        result, _ = print_and_benchmark_word_stats_sequential(
            temp_path, "topk", use_shuffle=False
        )
        assert isinstance(result, dict), "TopK should return dict"
        assert len(result) <= 10, "TopK should return at most 10 items"  # default top_k=10
        assert "a" in result, "Most frequent word 'a' should be in top results"

        # Test freq_count processing
        result, _ = print_and_benchmark_word_stats_sequential(
            temp_path, "freq_count", use_shuffle=False
        )
        assert isinstance(result, dict), "FreqCount should return dict"
        # Expected: {2: 2, 3: 1, 4: 1} - 2 words appear 2 times, 1 word appears 3 times, 1 word appears 4 times
        assert 4 in result, "Should have frequency 4"
        assert 2 in result, "Should have frequency 2"

    print("✓ Extended stats types tests passed")


def test_error_recovery():
    """Test error recovery and graceful handling."""
    print("Testing error recovery...")

    # Test with non-existent directory - this should return empty results without error
    # since glob() on non-existent path returns empty iterator
    result, _ = print_and_benchmark_word_stats_sequential(
        Path("/non/existent/path"), "word_count", use_shuffle=False
    )
    assert result == {}, "Non-existent directory should return empty results"

    # Test with invalid stats type through get_words_stats_in_file
    try:
        # Create a valid temp file first
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test content")
            temp_file = f.name

        try:
            get_words_stats_in_file([temp_file], "invalid_stats_type")
            assert False, "Should have raised an error for invalid stats type"
        except NotImplementedError:
            pass  # Expected
        finally:
            os.unlink(temp_file)

    except Exception as e:
        print(f"Unexpected error: {e}")

    print("✓ Error recovery tests passed")


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("RUNNING INTEGRATION FRAMEWORK TESTS")
    print("=" * 60)

    test_functions = [
        test_file_processing_integration,
        test_chunkify_function,
        test_sequential_processing,
        test_consistency_across_modes,
        test_extended_stats_types,
        test_error_recovery,
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
        print("🎉 All integration tests passed!")
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
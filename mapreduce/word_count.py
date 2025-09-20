"""
MapReduce Word Count Implementation

A demonstration of MapReduce concepts with sequential vs parallel processing.
Implements word counting across multiple text files using both approaches.

Problem: Implement a Word Count program that counts the frequency of each word in a large text dataset.

Input:
- One or more text files containing natural language text
- Files can be large (assume they might not fit in memory)

Output:
- A list of unique words and their frequencies
- Format: word: count (e.g., "hello: 5", "world: 3")
"""

# Standard library imports
import os
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Generator, Tuple 

def map_word_count(line: str) -> Generator[Tuple[str, int], None, None]:
    """
    Map phase: Extract words from a line and emit (word, 1) pairs.

    This function implements the "map" phase of MapReduce by processing a single
    line of text and emitting key-value pairs where each word is the key and
    the count (always 1) is the value.

    Args:
        line: Input text line to process

    Yields:
        Tuple of (word, count=1) for each word in the line

    Example:
        >>> list(map_word_count("hello world hello"))
        [('hello', 1), ('world', 1), ('hello', 1)]
    """
    words = line.split()
    for word in words:
        yield (word, 1)

def reduce_word_count(per_line_word_count: list[Generator[Tuple[str, int], None, None]]) -> defaultdict:
    """
    Reduce phase: Aggregate word counts from multiple generators.

    This function implements the "reduce" phase of MapReduce by collecting
    all (word, count) pairs from multiple generators and aggregating the
    counts for each unique word.

    Args:
        per_line_word_count: List of generators producing (word, count) pairs

    Returns:
        defaultdict with aggregated word counts for all processed lines

    Example:
        >>> gen1 = map_word_count("hello world")
        >>> gen2 = map_word_count("hello test")
        >>> result = reduce_word_count([gen1, gen2])
        >>> dict(result)
        {'hello': 2, 'world': 1, 'test': 1}
    """
    word_count = defaultdict(int)
    for gen in per_line_word_count:
        for word, count in gen:
            word_count[word] += count
    return word_count

def reduce_across_files(all_files_word_count: list[dict[str, int]]) -> dict[str, int]:
    """
    Final reduce phase: Aggregate word counts across multiple files.

    This function implements the final aggregation step in a distributed
    MapReduce system, combining word counts from multiple files into a
    single global word count dictionary.

    Args:
        all_files_word_count: List of dictionaries containing word counts from individual files

    Returns:
        Dictionary with global word counts across all processed files

    Example:
        >>> file1_counts = {'hello': 2, 'world': 1}
        >>> file2_counts = {'hello': 1, 'test': 3}
        >>> reduce_across_files([file1_counts, file2_counts])
        {'hello': 3, 'world': 1, 'test': 3}
    """
    word_count = defaultdict(int)
    for per_file_word_count in all_files_word_count:
        for word, count in per_file_word_count.items():
            word_count[word] += count
    return dict(word_count)

def count_words_in_file(file_name: str) -> defaultdict:
    """
    Process a single file and return word counts with timing information.

    This function coordinates the MapReduce process for a single file:
    1. Reads all lines from the file
    2. Applies map phase to each line
    3. Applies reduce phase to aggregate results
    4. Prints timing and debugging information

    Args:
        file_name: Path to the text file to process

    Returns:
        defaultdict containing word counts for the file

    Side Effects:
        Prints processing time and word count results to stdout
    """
    start_time = time.time()
    per_line_word_count = []
    with open(file_name, "r") as f:
        lines = f.readlines()
        for line in lines:
            per_line_word_count.append(map_word_count(line))

    word_count = reduce_word_count(per_line_word_count=per_line_word_count)
    end_time = time.time() - start_time
    print("-" * 20 + f"{file_name}" + "-" * 20)
    print(word_count)
    print(f"Processing {file_name} took {end_time:.4f} seconds")

    return word_count

def print_and_benchmark_word_count_sequential(data_dir: Path) -> Tuple[dict[str, int], float]:
    """
    Process all files sequentially and benchmark performance.

    This function implements sequential MapReduce processing where files
    are processed one after another in a single thread. This serves as
    a baseline for comparison with parallel processing.

    Args:
        data_dir: Path object pointing to directory containing .txt files

    Returns:
        Tuple containing:
        - Dictionary with aggregated word counts across all files
        - Float representing total processing time in seconds

    Side Effects:
        Prints detailed timing information and results to stdout
    """
    per_file_word_count: list[dict[str, int]] = []
    start_time = time.time()
    for file_path in data_dir.glob("*.txt"):
        per_file_word_count.append(count_words_in_file(file_name=str(file_path)))

    word_count = reduce_across_files(per_file_word_count)
    end_time = time.time() - start_time

    print("-" * 60)
    print("SEQUENTIAL MAPREDUCE RESULTS")
    print("-" * 60)
    print(word_count)
    print(f"Total time taken: {end_time:.4f} seconds")

    return word_count, end_time

def print_and_benchmark_word_count_parallel(data_dir: Path) -> Tuple[dict[str, int], float]:
    """
    Process all files in parallel using multiprocessing and benchmark performance.

    This function implements parallel MapReduce processing where files
    are processed simultaneously across multiple CPU cores. Each file
    is processed in a separate subprocess, enabling true parallelism.

    Args:
        data_dir: Path object pointing to directory containing .txt files

    Returns:
        Tuple containing:
        - Dictionary with aggregated word counts across all files
        - Float representing total processing time in seconds

    Side Effects:
        Prints detailed timing information and results to stdout
    """
    start_time = time.time()
    file_paths = [str(file_path) for file_path in data_dir.glob("*.txt")]

    # Use all available CPU cores for parallel processing
    with Pool(processes=os.cpu_count()) as pool:
        per_file_results = pool.map(count_words_in_file, file_paths)

    word_count = reduce_across_files(per_file_results)
    end_time = time.time() - start_time

    print("-" * 60)
    print("PARALLEL MAPREDUCE RESULTS")
    print("-" * 60)
    print(word_count)
    print(f"Total time taken: {end_time:.4f} seconds")
    print(f"Used {os.cpu_count()} CPU cores")

    return word_count, end_time


def calculate_speedup(sequential_time: float, parallel_time: float) -> float:
    """
    Calculate and display performance metrics for parallel vs sequential processing.

    This function analyzes the performance improvement achieved through
    parallel processing and calculates key metrics like speedup and efficiency.

    Args:
        sequential_time: Time taken for sequential processing in seconds
        parallel_time: Time taken for parallel processing in seconds

    Returns:
        Float representing the speedup factor (sequential_time / parallel_time)

    Side Effects:
        Prints detailed performance analysis to stdout
    """
    if parallel_time == 0:
        print("Warning: Parallel time is zero, cannot calculate speedup")
        return float('inf')

    speedup = sequential_time / parallel_time
    efficiency = speedup / os.cpu_count()

    print(f"\n{'=' * 60}")
    print("PERFORMANCE ANALYSIS")
    print(f"{'=' * 60}")
    print(f"Sequential time:     {sequential_time:.4f} seconds")
    print(f"Parallel time:       {parallel_time:.4f} seconds")
    print(f"Speedup:             {speedup:.2f}x")
    print(f"Efficiency:          {efficiency:.2f} ({efficiency*100:.1f}%)")
    print(f"CPU cores used:      {os.cpu_count()}")

    if speedup > 1:
        print(f"✓ Parallel processing is {speedup:.2f}x faster")
    else:
        print(f"⚠ Sequential processing is {1/speedup:.2f}x faster (overhead dominates)")

    print(f"{'=' * 60}")
    return speedup


# Test Functions
def test_map_word_count():
    """
    Test the map phase with known input.

    Verifies that the map_word_count function correctly splits text
    and emits (word, 1) pairs for each word.
    """
    result = list(map_word_count("hello world hello"))
    expected = [("hello", 1), ("world", 1), ("hello", 1)]
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ test_map_word_count passed")


def test_reduce_word_count():
    """
    Test the reduce phase aggregation.

    Verifies that the reduce_word_count function correctly aggregates
    word counts from multiple generators.
    """
    gen1 = map_word_count("hello world")
    gen2 = map_word_count("hello test")
    result = reduce_word_count([gen1, gen2])

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
    result = reduce_across_files([file1_counts, file2_counts])

    expected = {"hello": 3, "world": 1, "test": 3}
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ test_reduce_across_files passed")


def test_empty_input():
    """
    Test edge cases with empty input.

    Verifies that the system handles empty lines and files gracefully.
    """
    # Test empty line
    result = list(map_word_count(""))
    assert result == [], f"Expected empty list for empty input, got {result}"

    # Test whitespace-only line
    result = list(map_word_count("   \n\t  "))
    assert result == [], f"Expected empty list for whitespace input, got {result}"

    print("✓ test_empty_input passed")


def run_all_tests():
    """
    Run all test functions and report results.

    Executes all defined test functions and provides a summary
    of test results.
    """
    print("\n" + "=" * 40)
    print("RUNNING UNIT TESTS")
    print("=" * 40)

    test_functions = [
        test_map_word_count,
        test_reduce_word_count,
        test_reduce_across_files,
        test_empty_input
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
    else:
        print(f"⚠ {total - passed} tests failed")
    print("=" * 40)


if __name__ == "__main__":
    """
    Main execution block demonstrating MapReduce word counting.

    This script runs both sequential and parallel MapReduce implementations,
    compares their results for correctness, and analyzes performance differences.
    """
    # Run unit tests first
    run_all_tests()

    # Set up data directory
    data_dir = Path("./data")
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        print("Please create the data directory and add some .txt files")
        exit(1)

    txt_files = list(data_dir.glob("*.txt"))
    if not txt_files:
        print(f"Error: No .txt files found in {data_dir}")
        print("Please add some .txt files to process")
        exit(1)

    print(f"\nFound {len(txt_files)} files to process: {[f.name for f in txt_files]}")

    # Run sequential processing
    print("\n" + "=" * 60)
    print("SEQUENTIAL PROCESSING")
    print("=" * 60)
    sequential_word_count, sequential_time = print_and_benchmark_word_count_sequential(data_dir)

    # Run parallel processing
    print("\n" + "=" * 60)
    print("PARALLEL PROCESSING")
    print("=" * 60)
    parallel_word_count, parallel_time = print_and_benchmark_word_count_parallel(data_dir)

    # Verify correctness
    print("\n" + "=" * 60)
    print("CORRECTNESS VERIFICATION")
    print("=" * 60)
    try:
        assert sequential_word_count == parallel_word_count, "Results don't match!"
        print("✓ Sequential and parallel results are identical")
        print(f"✓ Total unique words processed: {len(sequential_word_count)}")
        print(f"✓ Total word instances: {sum(sequential_word_count.values())}")
    except AssertionError as e:
        print(f"✗ Correctness check failed: {e}")
        exit(1)

    # Performance analysis
    speedup = calculate_speedup(sequential_time, parallel_time)

    print("\n🎉 MapReduce word counting completed successfully!")

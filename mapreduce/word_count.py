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
import argparse
import os
import time
from collections import defaultdict
from functools import partial, reduce
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

def shuffle_results(per_line_word_count: list[Generator[Tuple[str, int], None, None]]) -> dict[str, list[int]]:
    per_word_dict = defaultdict(list)
    for gen in per_line_word_count: 
        for word, count in gen: 
            per_word_dict[word].append(count)
    
    return per_word_dict


def reduce_word_count(
    per_line_word_count: list[Generator[Tuple[str, int], None, None]],
) -> defaultdict:
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


def reduce_shuffled_word_count(shuffled_word_count: dict[str, list[int]]) -> dict[str, int]: 
    results = defaultdict(int)
    for word, count_list in shuffled_word_count.items(): 
        results[word] = reduce(lambda x, y: x + y , count_list)
    return results

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


def count_words_in_file(file_name: str, use_shuffle: bool = False) -> defaultdict:
    """
    Process a single file and return word counts with timing information.

    This function coordinates the MapReduce process for a single file:
    1. Reads all lines from the file
    2. Applies map phase to each line
    3. Applies reduce phase to aggregate results
    4. Prints timing and debugging information

    Args:
        file_name: Path to the text file to process
        use_shuffle: Whether to show explicit shuffle phase output

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

    # if use_shuffle:
    #     shuffled_results = shuffle_results(per_line_word_count)
    #     word_count = reduce_shuffled_word_count(shuffled_results)
    # else:
    word_count = reduce_word_count(per_line_word_count=per_line_word_count)
    end_time = time.time() - start_time
    
    print("-" * 20 + f"{file_name}" + "-" * 20)
    print(word_count)
    print(f"Processing {file_name} took {end_time:.4f} seconds")

    return word_count


def print_and_benchmark_word_count_sequential(
    data_dir: Path,
    use_shuffle: bool
) -> Tuple[dict[str, int], float]:
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
        per_file_word_count.append(count_words_in_file(file_name=str(file_path), use_shuffle=use_shuffle))

    word_count = reduce_across_files(per_file_word_count)
    end_time = time.time() - start_time

    print("-" * 60)
    print("SEQUENTIAL MAPREDUCE RESULTS")
    print("-" * 60)
    print(word_count)
    print(f"Total time taken: {end_time:.4f} seconds")

    return word_count, end_time


def print_and_benchmark_word_count_parallel(
    data_dir: Path,
    use_shuffle: bool
) -> Tuple[dict[str, int], float]:
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
    # Create a partial function that includes the use_shuffle parameter
    count_with_shuffle = partial(count_words_in_file, use_shuffle=use_shuffle)
    with Pool(processes=os.cpu_count()) as pool:
        per_file_results = pool.map(count_with_shuffle, file_paths)

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
        return float("inf")

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
        print(
            f"⚠ Sequential processing is {1/speedup:.2f}x faster (overhead dominates)"
        )

    print(f"{'=' * 60}")
    return speedup


def parse_arguments():
    """
    Parse command line arguments for MapReduce word count.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="MapReduce Word Count Implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python word_count.py                    # Run both sequential and parallel (default)
  python word_count.py sequential         # Run only sequential processing
  python word_count.py parallel           # Run only parallel processing
  python word_count.py both               # Run both sequential and parallel
  python word_count.py --shuffle          # Show shuffle phase output
  python word_count.py parallel --shuffle --data-dir ./level2_data  # Combined options
        """
    )

    parser.add_argument(
        "mode",
        nargs="?",
        choices=["sequential", "parallel", "both"],
        default="both",
        help="Processing mode: 'sequential', 'parallel', or 'both' (default: both)"
    )

    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Show explicit shuffle phase output for debugging and learning"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="./level2_data",
        help="Directory containing .txt files to process (default: ./data)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    """
    Main execution block demonstrating MapReduce word counting.

    This script runs both sequential and parallel MapReduce implementations,
    compares their results for correctness, and analyzes performance differences.
    """
    # Parse command line arguments
    args = parse_arguments()

    # Set up data directory
    data_dir = Path(args.data_dir)
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

    if args.shuffle:
        print("🔀 Shuffle phase visualization enabled")

    # Run sequential processing (if mode is 'sequential' or 'both')
    if args.mode in ["sequential", "both"]:
        print("\n" + "=" * 60)
        print("SEQUENTIAL PROCESSING")
        print("=" * 60)
        sequential_word_count, sequential_time = print_and_benchmark_word_count_sequential(
            data_dir,
            use_shuffle=args.shuffle
        )
    else:
        sequential_word_count = None
        sequential_time = None


    # Run parallel processing (if mode is 'parallel' or 'both')
    if args.mode in ["parallel", "both"]:
        print("\n" + "=" * 60)
        print("PARALLEL PROCESSING")
        print("=" * 60)
        parallel_word_count, parallel_time = print_and_benchmark_word_count_parallel(
            data_dir,
            use_shuffle=args.shuffle
        )
    else:
        parallel_word_count = None
        parallel_time = None

    # Verify correctness and analyze performance (only if both modes were run)
    if sequential_word_count is not None and parallel_word_count is not None:
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
    elif args.shuffle:
        # If only one mode was run with shuffle, provide summary
        result = sequential_word_count if sequential_word_count is not None else parallel_word_count
        print(f"\n📊 Processing completed with shuffle visualization")
        print(f"✓ Total unique words processed: {len(result)}")
        print(f"✓ Total word instances: {sum(result.values())}")

    print("\n🎉 MapReduce word counting completed successfully!")

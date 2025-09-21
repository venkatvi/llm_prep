"""
MapReduce Framework Implementation

A flexible demonstration of MapReduce concepts with multiple analysis types.
Supports word counting, character sum analysis, and average word length
calculation across multiple text files using both sequential and parallel
processing approaches.

Features:
- Multiple analysis types: word_count, sum_of_word_lengths, average_word_length
- Sequential and parallel processing modes
- Modular factory pattern for extensible operations
- Performance benchmarking and comparison
- Optional shuffle phase visualization
- Functional programming support with reduce operations
"""

# Standard library imports
import argparse
import os
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple

# Local imports
from factories.registry import (
    REDUCE_TYPE,
    get_mapreduce_class,
    reduce_across_files,
    reduce_shuffled_word_stats,
    shuffle_results,
)


def get_words_stats_in_file(
    file_names: list[str],
    stats_type: str,
    use_shuffle: bool = False,
    use_reduce: bool = False,
) -> REDUCE_TYPE:
    """Process multiple files with local combiner pattern and return aggregated
    statistics."""
    pid = os.getpid()
    mapreduce_class = get_mapreduce_class(stats_type)
    per_file_word_stats = []
    for file_name in file_names:
        start_time = time.time()
        per_line_word_stats = []
        with open(file_name, "r") as f:
            lines = f.readlines()
            for line in lines:
                per_line_word_stats.append(mapreduce_class.map(line))

        if use_shuffle:
            shuffled_results = shuffle_results(
                per_line_word_stats
            )  # always dict[str, int]
            # word_count: dict[word, count]
            # sum: dict[word, length] ? --> tuple[total characters, num_words]
            # means: tuple[total characters, num_words]
            current_file_word_stats = reduce_shuffled_word_stats(
                shuffled_results, use_reduce=use_reduce, stats_type=stats_type
            )
        else:
            # word_count: dict[word, count]
            # sum: dict[word, length] ? --> tuple[total characters, num_words]
            # means: tuple[total characters, num_words]
            current_file_word_stats = mapreduce_class.reduce(
                per_line_word_stats, use_reduce=use_reduce
            )
        end_time = time.time() - start_time

        print(f"Processing {file_name} in PID {pid} took {end_time:.12f} seconds")
        per_file_word_stats.append(current_file_word_stats)

    if len(file_names) > 1:
        # Local combiner: aggregate multiple files within one process
        modified_stats_type = mapreduce_class.get_modified_stats_type_for_local_combiner()
        modified_mapreduce_class = get_mapreduce_class(modified_stats_type)
        return modified_mapreduce_class.reduce_all(per_file_word_stats, use_reduce)
    else:
        return per_file_word_stats[0]


def print_and_benchmark_word_stats_sequential(
    data_dir: Path,
    stats_type: str,
    use_shuffle: bool,
    use_reduce: bool = False,
) -> Tuple[REDUCE_TYPE, float]:
    """Process all files sequentially and benchmark performance."""
    per_file_word_stats: list[dict[str, int]] = []
    start_time = time.time()
    for file_path in data_dir.glob("*.txt"):
        per_file_word_stats.append(
            get_words_stats_in_file(
                file_names=[str(file_path)],
                use_shuffle=use_shuffle,
                use_reduce=use_reduce,
                stats_type=stats_type,
            )
        )

    word_stats = reduce_across_files(
        per_file_word_stats, stats_type=stats_type, use_reduce=use_reduce
    )
    end_time = time.time() - start_time

    return word_stats, end_time


def chunkify(file_paths: list[str], num_processes: int) -> list[list[str]]:
    """Distribute files across processes for parallel processing."""
    num_files = len(file_paths)
    files_per_process = num_files // num_processes
    remainder = num_files - (num_processes * files_per_process)

    result = []
    start = 0
    for idx in range(num_processes):
        chunk_size = files_per_process + (1 if idx < remainder else 0)
        end = start + chunk_size

        if start < num_files:
            result.append(file_paths[start:end])
        start = end

    return result


def print_and_benchmark_word_stats_parallel(
    data_dir: Path,
    stats_type: str,
    use_shuffle: bool,
    use_reduce: bool = False,
    num_processes: int = None,
) -> Tuple[dict[str, int], float]:
    """Process all files in parallel using multiprocessing and benchmark
    performance."""
    start_time = time.time()
    file_paths = [str(file_path) for file_path in data_dir.glob("*.txt")]

    # Determine number of processes to use
    if num_processes is None:
        processes = os.cpu_count()
    else:
        processes = min(num_processes, os.cpu_count())

    file_paths = chunkify(file_paths, num_processes)
    # Create a partial function that includes the use_shuffle and use_reduce
    # parameters
    get_word_stats_with_params = partial(
        get_words_stats_in_file,
        use_shuffle=use_shuffle,
        use_reduce=use_reduce,
        stats_type=stats_type,
    )
    with Pool(processes=processes) as pool:
        per_file_results = pool.map(get_word_stats_with_params, file_paths)

    word_stats = reduce_across_files(
        per_file_results, stats_type=stats_type, use_reduce=use_reduce
    )
    end_time = time.time() - start_time

    print(f"Used {processes} CPU cores (out of {os.cpu_count()} available)")

    return word_stats, end_time


def calculate_speedup(sequential_time: float, parallel_time: float) -> float:
    """Calculate and display performance metrics for parallel vs sequential
    processing."""
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
    """Parse command line arguments for MapReduce word count."""
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
  python word_count.py --use-reduce       # Use functools.reduce instead of for loops
  python word_count.py --num-processes 2  # Use only 2 processes for parallel processing
  python word_count.py --stats-type sum_of_word_lengths  # Sum total character count
  python word_count.py --stats-type average_word_length  # Calculate average word length
  python word_count.py parallel --shuffle --data-dir ./level2_data  # Combined options
  python word_count.py --use-reduce --shuffle --stats-type sum_of_word_lengths  # All
        """,
    )

    parser.add_argument(
        "mode",
        nargs="?",
        choices=["sequential", "parallel", "both"],
        default="both",
        help="Processing mode: 'sequential', 'parallel', or 'both' " "(default: both)",
    )

    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Show explicit shuffle phase output for debugging and learning",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="./level2_data",
        help="Directory containing .txt files to process (default: ./data)",
    )

    parser.add_argument(
        "--use-reduce",
        action="store_true",
        help="Use functools.reduce instead of for loops for aggregation",
    )

    parser.add_argument(
        "--num-processes",
        type=int,
        default=2,
        help="Number of processes to use for parallel processing "
        "(default: all CPU cores)",
    )

    parser.add_argument(
        "--stats-type",
        type=str,
        choices=["word_count", "sum_of_word_lengths", "average_word_length", "topk", "freq_count"],
        default="word_count",
        help="Type of analysis to perform: 'word_count' (frequency), "
        "'sum_of_word_lengths' (total chars), 'average_word_length' "
        "(avg chars per word), topk, frequency_count (How many words occured with a frequency)",
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
        sequential_stats, sequential_time = print_and_benchmark_word_stats_sequential(
            data_dir,
            use_shuffle=args.shuffle,
            use_reduce=args.use_reduce,
            stats_type=args.stats_type,
        )
    else:
        sequential_stats = None
        sequential_time = None

    # Run parallel processing (if mode is 'parallel' or 'both')
    if args.mode in ["parallel", "both"]:
        print("\n" + "=" * 60)
        print("PARALLEL PROCESSING")
        print("=" * 60)
        parallel_stats, parallel_time = print_and_benchmark_word_stats_parallel(
            data_dir,
            use_shuffle=args.shuffle,
            use_reduce=args.use_reduce,
            num_processes=args.num_processes,
            stats_type=args.stats_type,
        )
    else:
        parallel_stats = None
        parallel_time = None

    # Verify correctness and analyze performance (only if both modes were run)
    if sequential_stats is not None and parallel_stats is not None:
        print("\n" + "=" * 60)
        print("CORRECTNESS VERIFICATION")
        print("=" * 60)
        try:
            assert sequential_stats == parallel_stats, "Results don't match!"
            print("-" * 60)
            print("MAPREDUCE RESULTS")
            print("-" * 60)
            print(parallel_stats)
            print("✓ Sequential and parallel results are identical")
        except AssertionError as e:
            print("-" * 60)
            print("PARALLEL MAPREDUCE RESULTS")
            print("-" * 60)
            print(parallel_stats)
            
            print("-" * 60)
            print("SEQUENTIAL MAPREDUCE RESULTS")
            print("-" * 60)
            print(sequential_stats)

            print(f"✗ Correctness check failed: {e}")
            exit(1)

        # Performance analysis
        speedup = calculate_speedup(sequential_time, parallel_time)
    elif args.shuffle:
        # If only one mode was run with shuffle, provide summary
        result = sequential_stats if sequential_stats is not None else parallel_stats
        print("\n📊 Processing completed with shuffle visualization")

    print("\n🎉 MapReduce word counting completed successfully!")

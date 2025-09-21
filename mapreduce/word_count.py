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
import itertools
import os
import time
from collections import defaultdict
from functools import partial, reduce
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Generator, Tuple, Union

REDUCE_TYPE = Union[
    dict[str, int], # word --> count 
    Tuple[int, int], # sum of lengths of words, # number of words 
    float, # average word length 
]
def get_map_function(stats_type: str) -> Callable:
    """Get the appropriate map function for the given stats type."""
    if stats_type == "word_count":
        return map_word_count
    elif stats_type == "sum_of_word_lengths":
        return map_word_length
    elif stats_type == "average_word_length":
        return map_word_length
    else:
        raise NotImplementedError()


def get_reduce_function(stats_type: str) -> Callable:
    """Get the appropriate reduce function for the given stats type."""
    if stats_type == "word_count":
        return reduce_word_count
    elif stats_type == "sum_of_word_lengths":
        return reduce_word_length
    elif stats_type == "average_word_length":
        return reduce_word_length_mean
    else:
        raise NotImplementedError()


def get_reduce_all_function(stats_type: str) -> Callable:
    """Get the appropriate reduce_all function for the given stats type."""
    if stats_type == "word_count":
        return reduce_all_word_counts
    elif stats_type == "sum_of_word_lengths":
        return reduce_all_word_length_sums
    elif stats_type == "average_word_length":
        return reduce_all_word_length_averages
    else:
        raise NotImplementedError(f"Unsupported stats_type: {stats_type}")


def map_word_length(line: str) -> Generator[Tuple[str, int], None, None]:
    """Map phase: Extract words from a line and emit (word, length) pairs."""
    words = line.split()
    for word in words:
        yield (word, len(word))

def map_word_count(line: str) -> Generator[Tuple[str, int], None, None]:
    """Map phase: Extract words from a line and emit (word, 1) pairs."""
    words = line.split()
    for word in words:
        yield (word, 1)


def shuffle_results(
    per_line_word_count: list[Generator[Tuple[str, int], None, None]],
) -> dict[str, list[int]]:
    """Shuffle phase: Group word counts by word across all generators."""
    per_word_dict = defaultdict(list)
    for gen in per_line_word_count:
        for word, count in gen:
            per_word_dict[word].append(count)

    return per_word_dict

def reduce_word_length(
    per_line_word_length: list[Generator[Tuple[str, int], None, None]],
    use_reduce: bool = False
) -> REDUCE_TYPE:
    """Reduce phase: Aggregate word length totals from multiple generators."""
    total_chars = 0
    num_words = 0

    for gen in per_line_word_length:
        for word, length in gen:
            total_chars += length
            num_words += 1

    return (total_chars, num_words)


def reduce_word_length_mean(
    per_line_word_length: list[Generator[Tuple[str, int], None, None]],
    use_reduce: bool = False
) -> REDUCE_TYPE:
    """Reduce phase: Aggregate word lengths for later average calculation."""
    total_chars = 0
    num_words = 0

    for gen in per_line_word_length:
        for word, length in gen:
            total_chars += length
            num_words += 1

    return (total_chars, num_words)


def reduce_word_count(
    per_line_word_count: list[Generator[Tuple[str, int], None, None]],
    use_reduce: bool = False,
) -> REDUCE_TYPE:
    """Reduce phase: Aggregate word counts from multiple generators."""
    word_count = defaultdict(int)
    if use_reduce:
        # In reduce function, first argument is the "reduced" datastructure which is carrying the results.
        def count_accumulator(
            accumulated_dict: defaultdict, word_count_tuple: Tuple[str, int]
        ):
            accumulated_dict[word_count_tuple[0]] += word_count_tuple[1]
            return accumulated_dict

        # flatten tuples
        all_tuples = itertools.chain(*per_line_word_count)  # returns an iterator

        # reduce(function, iteratable, initial_value), while the function's first argument is always reserved for the accumulated Datastructure
        word_count = reduce(count_accumulator, all_tuples, defaultdict(int))
        return word_count

    else:
        for gen in per_line_word_count:
            for word, count in gen:
                word_count[word] += count
    return word_count


def reduce_shuffled_word_stats(
    shuffled_word_stats: dict[str, list[int]], use_reduce: bool, stats_type: str
) -> REDUCE_TYPE:
    """Reduce phase after shuffle: Aggregate statistics for each word."""
    # unused stats_type, use_reduce
    if stats_type == "word_count":
        results = defaultdict(int)
        for word, count_list in shuffled_word_stats.items():
            results[word] = reduce(lambda x, y: x + y, count_list)
        return results
    elif stats_type == "sum_of_word_lengths" or stats_type=="average_word_length":
        num_words = 0
        total_character_count = 0
        for _, stats_list in shuffled_word_stats.items(): 
            num_words += len(stats_list) # number of times the word has occured
            total_character_count += sum(stats_list)
        return total_character_count, num_words
    else: 
        raise ValueError("Invalid stats type")




def reduce_all_word_counts(
    all_files_word_stats: list[dict[str, int]], use_reduce: bool = False
) -> dict[str, int]:
    """Aggregate word counts across multiple files."""
    word_stats = defaultdict(int)
    if use_reduce:
        def count_accumulator(
            accumulator_dict: defaultdict, input_dict: dict[str, int]
        ) -> defaultdict:
            for key, item in input_dict.items():
                accumulator_dict[key] += item
            return accumulator_dict

        word_stats = reduce(count_accumulator, all_files_word_stats, defaultdict(int))
    else:
        for per_file_word_stats in all_files_word_stats:
            for word, stats in per_file_word_stats.items():
                word_stats[word] += stats
    return dict(word_stats)


def reduce_all_word_length_sums(
    all_files_word_stats: list[tuple[int, int]], use_reduce: bool = False
) -> tuple[int, int]:
    """Aggregate word length sums across multiple files."""
    total_character_count = 0
    total_num_words = 0
    for per_file_word_stats in all_files_word_stats:
        character_count, num_words = per_file_word_stats
        total_character_count += character_count
        total_num_words += num_words
    return total_character_count, total_num_words


def reduce_all_word_length_averages(
    all_files_word_stats: list[tuple[int, int]], use_reduce:bool = False
) -> float:
    """Calculate global average word length across multiple files."""
    total_character_count = 0
    total_num_words = 0
    for per_file_word_stats in all_files_word_stats:
        character_count, num_words = per_file_word_stats
        total_character_count += character_count
        total_num_words += num_words
    return total_character_count / total_num_words if total_num_words > 0 else 0.0



def reduce_across_files(
    all_files_word_stats: list[REDUCE_TYPE], stats_type: str, use_reduce: bool = False
) -> REDUCE_TYPE:
    """Final reduce phase: Aggregate statistics across multiple files."""
    reduce_all_function = get_reduce_all_function(stats_type)
    return reduce_all_function(all_files_word_stats, use_reduce)


def get_words_stats_in_file(
    file_names: list[str], stats_type:str, use_shuffle: bool = False, use_reduce: bool = False,
) -> REDUCE_TYPE:
    """Process multiple files with local combiner pattern and return aggregated statistics."""
    pid = os.getpid()
    map_function = get_map_function(stats_type)
    reduce_function = get_reduce_function(stats_type)
    per_file_word_stats = []
    for file_name in file_names:
        print("!" * 20 + f"Processing {file_name} in process {pid}" + "!" * 20)
        start_time = time.time()
        per_line_word_stats = []
        with open(file_name, "r") as f:
            lines = f.readlines()
            for line in lines:
                per_line_word_stats.append(map_function(line))

        if use_shuffle:
            shuffled_results = shuffle_results(per_line_word_stats) # always dict[str, int]
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
            current_file_word_stats = reduce_function(
                per_line_word_stats, use_reduce=use_reduce
            )
        end_time = time.time() - start_time

        print("-" * 20 + f"{file_name}" + "-" * 20)
        print(current_file_word_stats)
        print(f"Processing {file_name} took {end_time:.4f} seconds")
        per_file_word_stats.append(current_file_word_stats)

    if len(file_names) > 1:
        # Local combiner: aggregate multiple files within one process
        if stats_type == "average_word_length":
            # For average, return totals (not average) for later global calculation
            reduce_all_function = get_reduce_all_function("sum_of_word_lengths")
            return reduce_all_function(per_file_word_stats)
        else:
            # Use factory pattern for consistency
            reduce_all_function = get_reduce_all_function(stats_type)
            if stats_type == "word_count":
                return reduce_all_function(per_file_word_stats, use_reduce)
            else:
                return reduce_all_function(per_file_word_stats)
    else:
        return per_file_word_stats[0]


def print_and_benchmark_word_stats_sequential(
    data_dir: Path, stats_type:str, use_shuffle: bool, use_reduce: bool = False,
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
                stats_type=stats_type
            )
        )
    
    word_stats = reduce_across_files(per_file_word_stats, stats_type=stats_type, use_reduce=use_reduce)
    end_time = time.time() - start_time

    print("-" * 60)
    print("SEQUENTIAL MAPREDUCE RESULTS")
    print("-" * 60)
    print(word_stats)
    print(f"Total time taken: {end_time:.4f} seconds")

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
    stats_type:str,
    use_shuffle: bool,
    use_reduce: bool = False,
    num_processes: int = None,
) -> Tuple[dict[str, int], float]:
    """Process all files in parallel using multiprocessing and benchmark performance."""
    start_time = time.time()
    file_paths = [str(file_path) for file_path in data_dir.glob("*.txt")]

    # Determine number of processes to use
    if num_processes is None:
        processes = os.cpu_count()
    else:
        processes = min(num_processes, os.cpu_count())

    file_paths = chunkify(file_paths, num_processes)
    # Create a partial function that includes the use_shuffle and use_reduce parameters
    get_word_stats_with_params = partial(
        get_words_stats_in_file, use_shuffle=use_shuffle, use_reduce=use_reduce, stats_type=stats_type
    )
    with Pool(processes=processes) as pool:
        per_file_results = pool.map(get_word_stats_with_params, file_paths)

    word_stats = reduce_across_files(per_file_results, stats_type=stats_type, use_reduce=use_reduce)
    end_time = time.time() - start_time

    print("-" * 60)
    print("PARALLEL MAPREDUCE RESULTS")
    print("-" * 60)
    print(word_stats)
    print(f"Total time taken: {end_time:.4f} seconds")
    print(f"Used {processes} CPU cores (out of {os.cpu_count()} available)")

    return word_stats, end_time


def calculate_speedup(sequential_time: float, parallel_time: float) -> float:
    """Calculate and display performance metrics for parallel vs sequential processing."""
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
  python word_count.py --use-reduce --shuffle --stats-type sum_of_word_lengths  # All features
        """,
    )

    parser.add_argument(
        "mode",
        nargs="?",
        choices=["sequential", "parallel", "both"],
        default="both",
        help="Processing mode: 'sequential', 'parallel', or 'both' (default: both)",
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
        help="Number of processes to use for parallel processing (default: all CPU cores)",
    )

    parser.add_argument(
        "--stats-type",
        type=str,
        choices=["word_count", "sum_of_word_lengths", "average_word_length"],
        default="word_count",
        help="Type of analysis to perform: 'word_count' (frequency), 'sum_of_word_lengths' (total chars), 'average_word_length' (avg chars per word)",
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
        sequential_stats, sequential_time = (
            print_and_benchmark_word_stats_sequential(
                data_dir, use_shuffle=args.shuffle, use_reduce=args.use_reduce, stats_type=args.stats_type
            )
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
            stats_type=args.stats_type
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
            print("✓ Sequential and parallel results are identical")
        except AssertionError as e:
            print(f"✗ Correctness check failed: {e}")
            exit(1)

        # Performance analysis
        speedup = calculate_speedup(sequential_time, parallel_time)
    elif args.shuffle:
        # If only one mode was run with shuffle, provide summary
        result = (
            sequential_stats
            if sequential_stats is not None
            else parallel_stats
        )
        print("\n📊 Processing completed with shuffle visualization")

    print("\n🎉 MapReduce word counting completed successfully!")

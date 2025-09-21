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
def get_map_function(stats_type:str ) -> Callable: 
    if stats_type == "word_count":
        return map_word_count
    elif stats_type == "sum_of_word_lengths": 
        return map_word_length 
    elif stats_type == "average_word_length": 
        return map_word_length 
    else: 
        raise NotImplementedError()
def get_reduce_function(stats_type:str ) -> Callable:
    if stats_type == "word_count":
        return reduce_word_count
    elif stats_type == "sum_of_word_lengths":
        return reduce_word_length
    elif stats_type == "average_word_length":
        return reduce_word_length_mean
    else:
        raise NotImplementedError()

def map_word_length(line: str) -> Generator[Tuple[str, int], None, None]: 
    words = line.split() 
    for word in words: 
        yield (word, len(word))

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


def shuffle_results(
    per_line_word_count: list[Generator[Tuple[str, int], None, None]],
) -> dict[str, list[int]]:
    """
    Shuffle phase: Group word counts by word across all generators.

    This function implements the "shuffle" phase of MapReduce by collecting
    all (word, count) pairs from multiple generators and grouping them by word.
    This creates the input needed for the reduce phase.

    Args:
        per_line_word_count: List of generators producing (word, count) pairs

    Returns:
        Dictionary mapping each word to a list of its counts from all generators

    Example:
        >>> gen1 = map_word_count("hello world")
        >>> gen2 = map_word_count("hello test")
        >>> shuffle_results([gen1, gen2])
        {'hello': [1, 1], 'world': [1], 'test': [1]}
    """
    per_word_dict = defaultdict(list)
    for gen in per_line_word_count:
        for word, count in gen:
            per_word_dict[word].append(count)

    return per_word_dict

def reduce_word_length(
    per_line_word_length: list[Generator[Tuple[str, int], None, None]],
    use_reduce: bool = False
) -> REDUCE_TYPE :
    # Return (total_characters, num_words) tuple for consistency
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
    # Return (total_chars, num_words) tuple - average calculated later
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
    """
    Reduce phase after shuffle: Aggregate counts for each word.

    This function takes the output of the shuffle phase and aggregates
    the counts for each word using functools.reduce.

    Args:
        shuffled_word_count: Dictionary mapping words to lists of counts
        use_reduce: Whether to use functools.reduce (currently always uses reduce)

    Returns:
        Dictionary with final aggregated word counts

    Example:
        >>> shuffled = {'hello': [1, 1], 'world': [1], 'test': [1]}
        >>> reduce_shuffled_word_count(shuffled, True)
        {'hello': 2, 'world': 1, 'test': 1}
    """
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
    """
    Aggregate word counts across multiple files.

    Args:
        all_files_word_stats: List of dictionaries containing word counts from individual files
        use_reduce: Whether to use functools.reduce for aggregation

    Returns:
        Dictionary with global word counts across all processed files

    Example:
        >>> file1_counts = {'hello': 2, 'world': 1}
        >>> file2_counts = {'hello': 1, 'test': 3}
        >>> reduce_all_word_counts([file1_counts, file2_counts])
        {'hello': 3, 'world': 1, 'test': 3}
    """
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
    """
    Aggregate word length sums across multiple files.

    Args:
        all_files_word_stats: List of (total_chars, num_words) tuples from individual files

    Returns:
        Tuple of (total_character_count, total_word_count) across all files

    Example:
        >>> file1_stats = (100, 20)  # 100 chars, 20 words
        >>> file2_stats = (150, 30)  # 150 chars, 30 words
        >>> reduce_all_word_length_sums([file1_stats, file2_stats])
        (250, 50)
    """
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
    """
    Calculate global average word length across multiple files.

    Args:
        all_files_word_stats: List of (total_chars, num_words) tuples from individual files

    Returns:
        Float representing average characters per word across all files

    Example:
        >>> file1_stats = (100, 20)  # avg = 5.0
        >>> file2_stats = (150, 30)  # avg = 5.0
        >>> reduce_all_word_length_averages([file1_stats, file2_stats])
        5.0
    """
    total_character_count = 0
    total_num_words = 0
    for per_file_word_stats in all_files_word_stats:
        character_count, num_words = per_file_word_stats
        total_character_count += character_count
        total_num_words += num_words
    return total_character_count / total_num_words if total_num_words > 0 else 0.0


def get_reduce_all_function(stats_type: str) -> Callable:
    """
    Get the appropriate reduce_all function for the given stats type.

    Args:
        stats_type: Type of analysis ('word_count', 'sum_of_word_lengths', 'average_word_length')

    Returns:
        Function that can aggregate statistics across multiple files

    Raises:
        NotImplementedError: If stats_type is not supported
    """
    if stats_type == "word_count":
        return reduce_all_word_counts
    elif stats_type == "sum_of_word_lengths":
        return reduce_all_word_length_sums
    elif stats_type == "average_word_length":
        return reduce_all_word_length_averages
    else:
        raise NotImplementedError(f"Unsupported stats_type: {stats_type}")


def reduce_across_files(
    all_files_word_stats: list[REDUCE_TYPE], stats_type: str, use_reduce: bool = False
) -> REDUCE_TYPE:
    """
    Final reduce phase: Aggregate statistics across multiple files.

    This function uses the factory pattern to delegate to the appropriate
    specialized reduce function based on stats_type.

    Args:
        all_files_word_stats: List of statistics from individual files
        stats_type: Type of analysis ('word_count', 'sum_of_word_lengths', 'average_word_length')
        use_reduce: Whether to use functools.reduce for aggregation (only used for word_count)

    Returns:
        Aggregated statistics across all processed files

    Raises:
        NotImplementedError: If stats_type is not supported
    """
    reduce_all_function = get_reduce_all_function(stats_type)

    return reduce_all_function(all_files_word_stats, use_reduce)


def get_words_stats_in_file(
    file_names: list[str], stats_type:str, use_shuffle: bool = False, use_reduce: bool = False, 
) -> REDUCE_TYPE:
    """
    Process multiple files with local aggregation and return combined word counts.

    This function implements a local combiner pattern for MapReduce processing,
    handling multiple files within a single process and performing local aggregation
    before returning results. This reduces the amount of data that needs to be
    shuffled and merged in the final global reduce phase.

    Processing Pipeline:
        1. For each file in the input list:
           a. Read all lines from the file into memory
           b. Apply map phase to each line (extract word-count pairs)
           c. Optional: Apply explicit shuffle phase for educational visualization
           d. Apply reduce phase to aggregate results for this file
           e. Report timing and results for this individual file
        2. If multiple files: Perform local aggregation across all file results
        3. If single file: Return the file result directly (optimization)

    Local Combiner Benefits:
        - Reduces network/IPC traffic in distributed systems
        - Minimizes memory usage in the final global reduce phase
        - Improves overall MapReduce performance when processes < files
        - Maintains same result correctness as individual file processing

    Args:
        file_names: List of file paths to process in this local combiner
        use_shuffle: If True, uses explicit shuffle phase (shuffle_results + reduce_shuffled_word_count)
                    If False, uses direct reduction (reduce_word_count)
        use_reduce: If True, uses functools.reduce for aggregation operations
                   If False, uses traditional for-loop based aggregation

    Returns:
        defaultdict[str, int]: Combined word counts across all processed files.
                              If single file, returns that file's word counts.
                              If multiple files, returns locally aggregated counts.

    Raises:
        FileNotFoundError: If any specified file does not exist
        PermissionError: If any file cannot be read due to permission restrictions

    Side Effects:
        - Prints process ID to show which process is handling the files
        - Prints separator line with file name for each file processed
        - Prints word count results to stdout for each individual file
        - Prints processing time in seconds for each individual file

    Performance Notes:
        - Loads entire files into memory (not suitable for very large individual files)
        - Generator-based map phase for memory efficiency during processing
        - Local aggregation reduces data volume for subsequent global reduce
        - Single file optimization eliminates unnecessary local reduce call

    Example:
        >>> # Single file (no local aggregation)
        >>> result = count_words_in_file(["data/small.txt"])
        !!!!!Processing data/small.txt in process 12345!!!!!
        --------------------data/small.txt--------------------
        defaultdict(<class 'int'>, {'hello': 3, 'world': 2})
        Processing data/small.txt took 0.0023 seconds

        >>> # Multiple files (with local aggregation)
        >>> result = count_words_in_file(["data/file1.txt", "data/file2.txt"], use_reduce=True)
        !!!!!Processing data/file1.txt in process 12345!!!!!
        [individual file results printed...]
        !!!!!Processing data/file2.txt in process 12345!!!!!
        [individual file results printed...]
        >>> # Returns locally combined results from both files
    """
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
    """
    Distribute files across processes for parallel processing.

    This function implements a load balancing algorithm that distributes files evenly
    across the specified number of processes. When files don't divide evenly, the
    remainder files are distributed to the first few processes, ensuring no process
    gets more than one extra file.

    Algorithm Details:
        1. Calculate base number of files per process (integer division)
        2. Calculate remainder files that need distribution
        3. Give first 'remainder' processes one extra file each
        4. Ensure all files are assigned and no duplicates exist

    Args:
        file_paths: List of file paths to be distributed across processes
        num_processes: Number of processes available for parallel processing

    Returns:
        list[list[str]]: List of chunks, where each chunk contains file paths
                        for one process to handle. Length equals num_processes
                        or fewer if there are fewer files than processes.

    Example:
        >>> files = ["f1.txt", "f2.txt", "f3.txt", "f4.txt", "f5.txt"]
        >>> chunkify(files, 2)
        [['f1.txt', 'f2.txt', 'f3.txt'], ['f4.txt', 'f5.txt']]

        >>> chunkify(files, 3)
        [['f1.txt', 'f2.txt'], ['f3.txt', 'f4.txt'], ['f5.txt']]

        >>> # More processes than files
        >>> chunkify(files, 10)
        [['f1.txt'], ['f2.txt'], ['f3.txt'], ['f4.txt'], ['f5.txt']]

    Performance Notes:
        - Time complexity: O(n) where n is number of processes
        - Space complexity: O(n) for the result list structure
        - All files are assigned exactly once (no duplicates or omissions)
    """
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
    """
    Process all files in parallel using multiprocessing and benchmark performance.

    This function implements parallel MapReduce processing where files
    are processed simultaneously across multiple CPU cores. Each file
    is processed in a separate subprocess, enabling true parallelism.

    Args:
        data_dir: Path object pointing to directory containing .txt files
        use_shuffle: If True, each file processing will use explicit shuffle phase
        use_reduce: If True, uses functools.reduce for all aggregation operations
        num_processes: Number of processes to use (default: None = all CPU cores)

    Returns:
        Tuple containing:
        - Dictionary with aggregated word counts across all files
        - Float representing total processing time in seconds

    Side Effects:
        Prints detailed timing information and results to stdout
    """
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

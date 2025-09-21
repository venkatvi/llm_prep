"""
MapReduce Class Registry

Central registry that provides MapReduce classes based on stats type.
Uses static methods from dedicated MapReduce classes for each analysis type.
"""

from typing import Union, Tuple
from collections import defaultdict

# Type definitions
REDUCE_TYPE = Union[
    dict[str, int],  # word --> count
    Tuple[int, int],  # sum of lengths of words, # number of words
    float,  # average word length
]


def get_mapreduce_class(stats_type: str):
    """Get the appropriate MapReduce class for the given stats type."""
    if stats_type == "word_count":
        from .word_count import WordCountMapReduce
        return WordCountMapReduce
    elif stats_type == "sum_of_word_lengths":
        from .word_length_sum import WordLengthSumMapReduce
        return WordLengthSumMapReduce
    elif stats_type == "average_word_length":
        from .word_length_average import WordLengthAverageMapReduce
        return WordLengthAverageMapReduce
    else:
        raise NotImplementedError(f"Unsupported stats_type: {stats_type}")


def shuffle_results(
    per_line_word_count,
) -> dict[str, list[int]]:
    """Shuffle phase: Group word counts by word across all generators."""
    per_word_dict = defaultdict(list)
    for gen in per_line_word_count:
        for word, count in gen:
            per_word_dict[word].append(count)

    return per_word_dict


def reduce_shuffled_word_stats(
    shuffled_word_stats: dict[str, list[int]], use_reduce: bool, stats_type: str
) -> REDUCE_TYPE:
    """Reduce phase after shuffle: Aggregate statistics for each word."""
    mapreduce_class = get_mapreduce_class(stats_type)
    return mapreduce_class.reduce_shuffled(shuffled_word_stats)


def reduce_across_files(
    all_files_word_stats: list[REDUCE_TYPE], stats_type: str, use_reduce: bool = False
) -> REDUCE_TYPE:
    """Final reduce phase: Aggregate statistics across multiple files."""
    mapreduce_class = get_mapreduce_class(stats_type)
    return mapreduce_class.reduce_all(all_files_word_stats, use_reduce)
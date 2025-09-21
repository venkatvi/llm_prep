"""
Word Count MapReduce Operations

Implements MapReduce operations for word frequency analysis.
Provides static methods for all phases of word counting pipeline:
map, reduce, reduce_all, and reduce_shuffled operations.
"""

import itertools
from collections import defaultdict
from functools import reduce
from typing import Generator, Tuple


class WordCountMapReduce:
    """MapReduce operations for word frequency counting."""

    @staticmethod
    def map(line: str) -> Generator[Tuple[str, int], None, None]:
        """Map phase: Extract words from a line and emit (word, 1) pairs."""
        words = line.split()
        for word in words:
            yield (word, 1)

    @staticmethod
    def reduce(
        per_line_word_count: list[Generator[Tuple[str, int], None, None]],
        use_reduce: bool = False,
    ) -> dict[str, int]:
        """Reduce phase: Aggregate word counts from multiple generators."""
        word_count = defaultdict(int)
        if use_reduce:
            # In reduce function, first argument is the "reduced" datastructure
            # which is carrying the results.
            def count_accumulator(
                accumulated_dict: defaultdict, word_count_tuple: Tuple[str, int]
            ):
                accumulated_dict[word_count_tuple[0]] += word_count_tuple[1]
                return accumulated_dict

            # flatten tuples
            all_tuples = itertools.chain(*per_line_word_count)  # returns an iterator

            # reduce(function, iteratable, initial_value), while the function's
            # first argument is always reserved for the accumulated Datastructure
            word_count = reduce(count_accumulator, all_tuples, defaultdict(int))
            return word_count

        else:
            for gen in per_line_word_count:
                for word, count in gen:
                    word_count[word] += count
        return word_count

    @staticmethod
    def reduce_all(
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

            word_stats = reduce(
                count_accumulator, all_files_word_stats, defaultdict(int)
            )
        else:
            for per_file_word_stats in all_files_word_stats:
                for word, stats in per_file_word_stats.items():
                    word_stats[word] += stats
        return dict(word_stats)

    @staticmethod
    def reduce_shuffled(shuffled_word_stats: dict[str, list[int]]) -> dict[str, int]:
        """Reduce word counts after shuffle phase."""
        results = defaultdict(int)
        for word, count_list in shuffled_word_stats.items():
            results[word] = reduce(lambda x, y: x + y, count_list)
        return results

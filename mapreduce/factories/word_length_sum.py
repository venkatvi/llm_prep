"""
Word Length Sum MapReduce Operations

Implements MapReduce operations for total character count analysis.
Provides static methods for all phases of character counting pipeline:
map, reduce, reduce_all, and reduce_shuffled operations.
"""

from typing import Generator, Tuple


class WordLengthSumMapReduce:
    """MapReduce operations for calculating total character count."""

    @staticmethod
    def map(line: str) -> Generator[Tuple[str, int], None, None]:
        """Map phase: Extract words from a line and emit (word, length) pairs."""
        words = line.split()
        for word in words:
            yield (word, len(word))

    @staticmethod
    def reduce(
        per_line_word_length: list[Generator[Tuple[str, int], None, None]],
        use_reduce: bool = False,
    ) -> Tuple[int, int]:
        """Reduce phase: Aggregate word length totals from multiple generators."""
        total_chars = 0
        num_words = 0

        for gen in per_line_word_length:
            for word, length in gen:
                total_chars += length
                num_words += 1

        return (total_chars, num_words)

    @staticmethod
    def reduce_all(
        all_files_word_stats: list[Tuple[int, int]], use_reduce: bool = False
    ) -> Tuple[int, int]:
        """Aggregate word length sums across multiple files."""
        total_character_count = 0
        total_num_words = 0
        for per_file_word_stats in all_files_word_stats:
            character_count, num_words = per_file_word_stats
            total_character_count += character_count
            total_num_words += num_words
        return total_character_count, total_num_words

    @staticmethod
    def reduce_shuffled(shuffled_word_stats: dict[str, list[int]]) -> Tuple[int, int]:
        """Reduce word lengths after shuffle phase."""
        num_words = 0
        total_character_count = 0
        for _, stats_list in shuffled_word_stats.items():
            num_words += len(stats_list)  # number of times the word has occurred
            total_character_count += sum(stats_list)
        return total_character_count, num_words

"""
MapReduce Top-K Word Frequency Implementation

This module demonstrates computing the top-K most frequent words across multiple
documents using the MapReduce paradigm. It follows the classic word count pattern
but adds a final sorting and selection step to find the most frequent words.

Example:
    documents = [("doc1", "the cat sat"), ("doc2", "the dog ran")]
    k = 2
    Expected: [("the", 2), ("cat", 1)] or similar based on frequency

The process follows MapReduce phases:
1. Map: Extract words and emit (word, 1) pairs
2. Shuffle: Group counts by word
3. Reduce: Sum counts and select top-K words by frequency
"""

import heapq
import re
from collections import defaultdict
from typing import Generator, Tuple, List, Dict


def map_words(doc: Tuple[str, str]) -> Generator[Tuple[str, int], None, None]:
    """
    Map function: extract words from a document and emit word counts.

    Processes one document at a time and yields (word, 1) pairs for each
    word found in the document text. Includes basic text preprocessing
    for better word extraction.

    Args:
        doc: Tuple of (document_name, document_text)

    Yields:
        Tuple of (word, 1) for each word in the document

    Example:
        >>> list(map_words(("doc1", "Hello, world!")))
        [('hello', 1), ('world', 1)]
    """
    doc_name, text = doc
    print(f"Mapping doc {doc_name}")

    # Enhanced text processing: lowercase and extract words only
    words = re.findall(r'\b\w+\b', text.lower())
    for word in words:
        yield (word, 1)


def shuffle_words(
    mapped_words: List[Generator[Tuple[str, int], None, None]]
) -> Dict[str, List[int]]:
    """
    Shuffle function: group word counts by word.

    Collects all word count pairs from the map phase and groups them
    by word for the reduce phase.

    Args:
        mapped_words: List of generators from map phase

    Returns:
        Dictionary mapping words to lists of their counts
    """
    shuffled_results = defaultdict(list)
    for gen in mapped_words:
        for word, count in gen:
            shuffled_results[word].append(count)
    return dict(shuffled_results)


def reduce_topk(
    shuffled_results: Dict[str, List[int]], topk: int
) -> List[Tuple[str, int]]:
    """
    Reduce function: sum word counts and select top-K most frequent.

    Aggregates all counts for each word and uses heap-based selection
    for efficient top-K computation, especially beneficial for large datasets.

    Args:
        shuffled_results: Dictionary mapping words to lists of counts
        topk: Number of top words to return

    Returns:
        List of (word, total_count) tuples sorted by frequency (descending)

    Example:
        >>> reduce_topk({"cat": [1, 1], "dog": [1]}, 2)
        [('cat', 2), ('dog', 1)]
    """
    # Aggregate counts efficiently
    word_counts = {
        word: sum(counts) for word, counts in shuffled_results.items()
    }

    # Use heap for efficient top-K selection: O(n log k) vs O(n log n)
    return heapq.nlargest(topk, word_counts.items(), key=lambda x: x[1])


def get_topk(
    documents: List[Tuple[str, str]], k: int
) -> List[Tuple[str, int]]:
    """
    Main function: compute top-K words using MapReduce pattern.

    Orchestrates the complete MapReduce pipeline for finding the most
    frequent words across all documents.

    Args:
        documents: List of (document_name, document_text) tuples
        k: Number of top words to return

    Returns:
        List of (word, count) tuples for the k most frequent words

    Example:
        >>> docs = [("d1", "cat dog"), ("d2", "cat bird")]
        >>> get_topk(docs, 2)
        [('cat', 2), ('dog', 1)]
    """
    # Map phase: process each document
    mapped_words = [map_words(doc) for doc in documents]

    # Shuffle phase: group by word
    shuffled_words = shuffle_words(mapped_words)

    # Reduce phase: aggregate and select top-K
    return reduce_topk(shuffled_words, k)

if __name__ == "__main__":
    """Example usage and testing of Top-K word frequency computation."""
    # Test case
    documents = [
        ("doc1", "the cat sat on the mat"),
        ("doc2", "the dog ran in the park"),
        ("doc3", "cat and dog are pets"),
    ]
    k = 3

    print("Top-K Word Frequency using MapReduce")
    print("=" * 40)
    print("Documents:")
    for doc_name, content in documents:
        print(f"  {doc_name}: '{content}'")
    print()

    # Expected top 3: [("the", 4), ("cat", 2), ("dog", 2)]
    topk_results = get_topk(documents, k)

    print(f"Top {k} most frequent words:")
    for i, (word, count) in enumerate(topk_results, 1):
        print(f"  {i}. '{word}': {count} occurrences")

    print(f"\nRaw output: {topk_results}")
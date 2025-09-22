
"""
MapReduce Practice Problem: Computing Average Scores

This module demonstrates two approaches to computing average scores per user:
1. Brute force approach (direct aggregation)
2. Classic MapReduce pattern (map -> shuffle -> reduce)

The MapReduce approach follows the standard distributed computing paradigm
where data is processed in parallel map tasks, shuffled by key, and then
reduced to final results.
"""

from typing import Tuple, Generator, List, Dict
from collections import defaultdict


def aggregate_per_user_scores(
    scores: List[Tuple[str, int]]
) -> Dict[str, List[int]]:
    """
    Aggregate scores by user name using direct approach.

    Args:
        scores: List of (user_name, score) tuples

    Returns:
        Dictionary mapping user names to lists of their scores
    """
    mapped_scores = defaultdict(list)
    for score in scores:
        name, val = score
        mapped_scores[name].append(val)
    return dict(mapped_scores)


def compute_average_per_user(
    mapped_scores: Dict[str, List[int]]
) -> List[Tuple[str, float]]:
    """
    Compute average score for each user from pre-grouped scores.

    Args:
        mapped_scores: Dictionary mapping user names to score lists

    Returns:
        List of (user_name, average_score) tuples
    """
    average_scores = []
    for key, val in mapped_scores.items():
        all_values = mapped_scores[key]
        average_val = sum(all_values) / len(all_values)
        average_scores.append((key, average_val))
    return average_scores


def compute_average_bruteforce(
    scores: List[Tuple[str, int]]
) -> List[Tuple[str, float]]:
    """
    Compute average scores using brute force approach.

    This approach directly aggregates scores by user and computes averages
    without following the MapReduce pattern.

    Args:
        scores: List of (user_name, score) tuples

    Returns:
        List of (user_name, average_score) tuples
    """
    # Dict[str, List[int]]
    per_user_mapped_scores = aggregate_per_user_scores(scores)
    # List[Tuple[str, float]]
    average_scores = compute_average_per_user(per_user_mapped_scores)
    return average_scores


def map_score(score: Tuple[str, int]) -> Generator[Tuple[str, int], None, None]:
    """
    MapReduce Map function: process one score record.

    Classic MapReduce mapper that processes one entry at a time and yields
    a key-value pair for the shuffle phase.

    Args:
        score: A single (user_name, score) tuple

    Yields:
        (user_name, score) tuple for shuffle phase
    """
    yield (score[0], score[1])


def shuffle_scores(
    mapped_results: List[Generator[Tuple[str, int], None, None]]
) -> Dict[str, List[int]]:
    """
    MapReduce Shuffle function: group mapped results by key.

    Collects all mapped results and groups them by user name for the
    reduce phase.

    Args:
        mapped_results: List of generators from map phase

    Returns:
        Dictionary mapping user names to lists of their scores
    """
    shuffled_results = defaultdict(list)
    for gen in mapped_results:  # Each generator from map phase
        for name, val in gen:  # Extract values from generator
            shuffled_results[name].append(val)
    return dict(shuffled_results)


def reduce_score(key: str, values: List[int]) -> Tuple[str, float]:
    """
    MapReduce Reduce function: compute average for one user.

    Args:
        key: User name
        values: List of all scores for this user

    Returns:
        (user_name, average_score) tuple
    """
    return (key, sum(values) / len(values))


def compute_average_mr(
    scores: List[Tuple[str, int]]
) -> List[Tuple[str, float]]:
    """
    Compute average scores using MapReduce pattern.

    Follows the classic MapReduce paradigm:
    1. Map: Process each score individually
    2. Shuffle: Group scores by user name
    3. Reduce: Compute average for each user

    Args:
        scores: List of (user_name, score) tuples

    Returns:
        List of (user_name, average_score) tuples
    """
    # Map phase: apply map function to each score
    mapped_results = [map_score(score) for score in scores]

    # Shuffle phase: group by user name
    shuffled_scores = shuffle_scores(mapped_results)

    # Reduce phase: compute average for each user
    all_tuples = shuffled_scores.items()  # List of (key, values) tuples
    return [reduce_score(v[0], v[1]) for v in all_tuples]

if __name__ == "__main__":
    """Example usage and comparison of both approaches."""
    # List[Tuple[str, int]]
    scores = [
        ("alice", 85),
        ("bob", 90),
        ("alice", 92),
        ("bob", 88),
        ("charlie", 95),
    ]

    print("Brute Force Approach:")
    average_scores = compute_average_bruteforce(scores)
    print(f"Input: {scores}")
    print(f"Output: {average_scores}")

    print("\nMapReduce Approach:")
    average_scores = compute_average_mr(scores)
    print(f"Input: {scores}")
    print(f"Output: {average_scores}")
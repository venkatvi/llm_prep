""" 
# Test data
weather_data = [
    ("NYC", 2020, 25), ("NYC", 2020, 30), ("NYC", 2019, 20),
    ("LA", 2020, 35), ("LA", 2019, 32)
]
# Expected: sorted by year, then by temperature within each year

"""
from collections import defaultdict
from typing import Generator, Tuple 

class CompositeKey:
    """Composite key for secondary sorting by year, then temperature."""

    def __init__(self, year: int, temperature: int):
        self.year = year
        self.temperature = temperature

    def __lt__(self, other: 'CompositeKey') -> bool:
        """Sort by year first, then by temperature within same year."""
        if self.year != other.year:
            return self.year < other.year
        return self.temperature < other.temperature

    def __eq__(self, other: 'CompositeKey') -> bool:
        """Check equality based on both year and temperature."""
        return self.year == other.year and self.temperature == other.temperature

    def __hash__(self) -> int:
        """Enable use as dictionary key."""
        return hash((self.year, self.temperature))

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"CompositeKey(year={self.year}, temp={self.temperature})"

def secondary_sort_mapper(record: Tuple[str, int, int]) -> Generator[Tuple[Tuple[str, CompositeKey], Tuple[str, int, int]], None, None]:
    """
    Map function: Create composite key for secondary sorting.

    Input: (location, year, temperature)
    Output: ((location, CompositeKey(year, temp)), original_record)

    The composite key enables sorting by location (primary), then year, then temperature.
    """
    location, year, temp = record
    composite_key = (location, CompositeKey(year, temp))  # Primary + secondary key
    yield (composite_key, record)

def secondary_sort_reducer(key: str, records: list[Tuple[CompositeKey, Tuple[str, int, int]]]) -> list[Tuple[CompositeKey, Tuple[str, int, int]]]:
    """
    Reduce function: Pass through sorted records.

    In secondary sort, the reducer is simple because sorting happens in shuffle phase.
    Records are already sorted by CompositeKey (year, then temperature).

    Args:
        key: Location (primary key)
        records: List of (CompositeKey, original_record) tuples, pre-sorted

    Returns:
        Same records, maintaining sort order
    """
    return records

def secondary_sort_shuffle(mapped_results: list[Generator[Tuple[Tuple[str, CompositeKey], Tuple[str, int, int]], None, None]]) -> dict[str, list[Tuple[CompositeKey, Tuple[str, int, int]]]]:
    """
    Shuffle function: Group by location and sort by CompositeKey.

    This is where secondary sort happens! Records are grouped by location
    and then sorted by CompositeKey (year, then temperature within year).

    Args:
        mapped_results: List of generators from map phase

    Returns:
        Dictionary mapping location to sorted list of (CompositeKey, record) tuples
    """
    per_location_dict = defaultdict(list)

    # Group records by location
    for gen in mapped_results:
        for composite_key, record in gen:
            location, ck = composite_key
            per_location_dict[location].append((ck, record))

    # Sort each location's records by CompositeKey (year, then temperature)
    for location, records in per_location_dict.items():
        per_location_dict[location] = sorted(records, key=lambda x: x[0])

    return dict(per_location_dict)

# Test data
weather_data = [
    ("NYC", 2020, 25), ("NYC", 2020, 30), ("NYC", 2019, 20),
    ("LA", 2020, 35), ("LA", 2019, 32)
]
# Expected: sorted by year, then by temperature within each year

mapped_results = [secondary_sort_mapper(record) for record in weather_data]
shuffled_dict = secondary_sort_shuffle(mapped_results)

results = []
for k, v in shuffled_dict.items():
    records_per_location = secondary_sort_reducer(k, v)
    for record in records_per_location: 
        results.append(record[1])

print(results)
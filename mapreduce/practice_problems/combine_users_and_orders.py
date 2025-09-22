"""
MapReduce Join Implementation: Combining Users and Orders

This module demonstrates how to implement a join operation using the MapReduce
paradigm. It joins user data with order data based on user_id, following the
classic reduce-side join pattern used in distributed computing systems.

Example:
    users = [("users", (1, "Alice")), ("users", (2, "Bob"))]
    orders = [("orders", (101, 1, 250)), ("orders", (102, 1, 150))]

    Expected output:
    [(1, ("Alice", 101, 250)), (1, ("Alice", 102, 150)), (2, ("Bob", 103, 300))]

The join is performed by:
1. Map: Emit records with user_id as key, tagged by source table
2. Shuffle: Group all records by user_id
3. Reduce: Join user data with their orders for each user_id
"""

from collections import defaultdict
from typing import Generator, Tuple, Union, List, Dict


def map_users(
    user_record: Tuple[str, Tuple[int, str]]
) -> Generator[Tuple[int, Tuple[str, str]], None, None]:
    """
    Map function for user records.

    Extracts user_id as the join key and tags the record with source table.

    Args:
        user_record: Tuple of (table_name, (user_id, user_name))

    Yields:
        Tuple of (user_id, (table_name, user_name))
    """
    table_name, user_entry = user_record
    user_id, user_name = user_entry
    yield (user_id, (table_name, user_name))


def map_orders(
    order_record: Tuple[str, Tuple[int, int, int]]
) -> Generator[Tuple[int, Tuple[str, int, int]], None, None]:
    """
    Map function for order records.

    Extracts user_id as the join key and tags the record with source table.

    Args:
        order_record: Tuple of (table_name, (order_id, user_id, amount))

    Yields:
        Tuple of (user_id, (table_name, order_id, amount))
    """
    table_name, order_entry = order_record
    order_id, user_id, amount = order_entry
    yield (user_id, (table_name, order_id, amount))


def shuffle_results(
    mapped_users: List[Generator[Tuple[int, Tuple[str, str]], None, None]],
    mapped_orders: List[Generator[Tuple[int, Tuple[str, int, int]], None, None]]
) -> Dict[int, List[Union[Tuple[str, str], Tuple[str, int, int]]]]:
    """
    Shuffle function: group mapped results by user_id.

    Collects all mapped results from both users and orders and groups them
    by user_id for the reduce phase.

    Args:
        mapped_users: List of generators from user map phase
        mapped_orders: List of generators from order map phase

    Returns:
        Dictionary mapping user_id to list of tagged records
    """
    per_user_dict = defaultdict(list)

    # Collect user records
    for gen_user in mapped_users:
        for user_id, table_mapping in gen_user:
            per_user_dict[user_id].append(table_mapping)

    # Collect order records
    for gen_order in mapped_orders:
        for user_id, order_mapping in gen_order:
            per_user_dict[user_id].append(order_mapping)

    return dict(per_user_dict)


def reduce_on_user_id(
    key: int,
    values: List[Union[Tuple[str, str], Tuple[str, int, int]]]
) -> List[Tuple[int, Tuple[str, int, int]]]:
    """
    Reduce function: join user data with orders for one user_id.

    Performs the actual join operation by combining user information with
    all orders for that user. Uses a two-phase approach to handle records
    in any order.

    Args:
        key: user_id (the join key)
        values: List of tagged records for this user from both tables

    Returns:
        List of joined records: (user_id, (user_name, order_id, amount))
    """
    user_name = ""
    orders = []

    # Phase 1: Separate users from orders by table tag
    for value in values:
        if value[0] == "orders":
            orders.append((value[1], value[2]))  # (order_id, amount)
        elif value[0] == "users":
            user_name = value[1]

    # Phase 2: Create joined results (Cartesian product)
    results = []
    for order_id, amount in orders:
        results.append((key, (user_name, order_id, amount)))

    return results


def join_tables(
    users: List[Tuple[str, Tuple[int, str]]],
    orders: List[Tuple[str, Tuple[int, int, int]]]
) -> List[Tuple[int, Tuple[str, int, int]]]:
    """
    Join users and orders using MapReduce pattern.

    Implements a complete reduce-side join following the MapReduce paradigm:
    1. Map phase: Process users and orders separately
    2. Shuffle phase: Group by user_id
    3. Reduce phase: Join user data with their orders

    Args:
        users: List of user records (table_name, (user_id, user_name))
        orders: List of order records (table_name, (order_id, user_id, amount))

    Returns:
        List of joined records: (user_id, (user_name, order_id, amount))
    """
    # Map phase: apply map functions to each record
    mapped_users = [map_users(user) for user in users]
    mapped_orders = [map_orders(order) for order in orders]

    # Shuffle phase: group by user_id
    shuffled_results_per_user_id = shuffle_results(mapped_users, mapped_orders)

    # Reduce phase: join and flatten results
    joined_results = []
    for k, v in shuffled_results_per_user_id.items():
        joined_results.extend(reduce_on_user_id(k, v))

    return joined_results


if __name__ == "__main__":
    """Example usage and testing of the MapReduce join implementation."""
    # Test data
    users = [("users", (1, "Alice")), ("users", (2, "Bob"))]
    orders = [
        ("orders", (101, 1, 250)),
        ("orders", (102, 1, 150)),
        ("orders", (103, 2, 300)),
    ]

    print("MapReduce Join Implementation")
    print("=" * 40)
    print(f"Users: {users}")
    print(f"Orders: {orders}")
    print()

    # Expected output:
    # [(1, ("Alice", 101, 250)), (1, ("Alice", 102, 150)), (2, ("Bob", 103, 300))]
    joined_table = join_tables(users, orders)
    print("Joined Results:")
    for result in joined_table:
        user_id, (user_name, order_id, amount) = result
        print(f"  User {user_id} ({user_name}) -> Order {order_id}: ${amount}")

    print(f"\nRaw output: {joined_table}")
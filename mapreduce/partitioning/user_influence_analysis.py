"""
Challenge 1: User Influence Score with Partitioning Strategies

This is a starter template that demonstrates how to approach the partitioning
and skew handling challenges. Your task is to implement different partitioning
strategies and measure their effectiveness.

Problem: Calculate user influence score based on activity count, but handle
the fact that power users create extreme data skew.
"""

import json
import math
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Generator
from dataclasses import dataclass
from functools import reduce
import itertools
import hashlib 
import random 
        


@dataclass
class PartitioningStrategy:
    """Configuration for a partitioning strategy."""
    name: str
    partition_func: callable
    num_partitions: int


class UserInfluenceMapReduce:
    """
    MapReduce implementation for calculating user influence scores
    with different partitioning strategies.

    """

    @staticmethod
    def map(line: str) -> Generator[Tuple[str, int], None, None]:
        """
        Map phase: Extract user_id from activity records.

        Parse JSON line and extract user_id, emit (user_id, 1) for counting
        """
        record = json.loads(line.strip())
        user_id = str(record["user_id"])
        yield(user_id, 1)

    @staticmethod
    def reduce(per_line_user_activities: List[Generator], use_reduce: bool = False) -> Dict[str, int]:
        """
        Reduce phase: Count activities per user.

        Aggregate user activity counts
        """
        activities_count_per_user = defaultdict(int)
        if use_reduce:
            def count_accumulator(acc_dict: defaultdict, user_activity:Tuple[str, int]) -> defaultdict: 
                user_id, count = user_activity
                acc_dict[user_id] += count 
                return acc_dict 
            
            all_tuples = itertools.chain(*per_line_user_activities)
            activities_count_per_user = reduce(count_accumulator, all_tuples, defaultdict(int))
        else: 
            for gen in per_line_user_activities: 
                userid, count = gen
                activities_count_per_user[userid] += count

        return dict(activities_count_per_user)

    @staticmethod
    def partition_aware_map(line: str, partitioning_strategy: PartitioningStrategy) -> Generator[Tuple[int, Tuple[str, int]], None, None]:
        """
        Map phase that outputs partition-aware key-value pairs.

        Parse line, determine partition, emit (partition_id, (user_id, 1))
        """
        # 1. Parse JSON to get user data
        record = json.loads(line) # dict 
        user_id = str(record["user_id"])
        # 2. Use partitioning_strategy.partition_func to determine partition
        partition_id = partitioning_strategy.partition_func(record, partitioning_strategy.num_partitions)
        # 3. Yield (partition_id, (user_id, 1))
        yield(partition_id, (user_id, 1))

    @staticmethod
    def partition_aware_reduce(partitioned_data: Dict[int, List[Tuple[str, int]]]) -> Dict[int, Dict[str, int]]:
        """
        Reduce phase that processes each partition separately.

        Process each partition's data and return per-partition results
        """
        # 1. For each partition, aggregate user counts
        def reduce_partition(data: List[Tuple[str, int]]) -> Dict[str, int]:
            def count_accumulator(acc_dict: defaultdict, user_count_tuple: Tuple[str, int]) -> defaultdict: 
                userid, count = user_count_tuple
                acc_dict[userid] += count 
                return acc_dict  
            return dict(reduce(count_accumulator, data, defaultdict(int)))
        # 2. Return {partition_id: {user_id: count, ...}, ...}
        return {partition_id: reduce_partition(tuples_list) for partition_id, tuples_list in partitioned_data.items()}
        

def hash_partition(user_data: Dict, num_partitions: int) -> int:
    """
    Simple hash-based partitioning.

    Implements hash partitioning strategy
    """
    user_id = str(user_data["user_id"])
    hash_obj = hashlib.md5(user_id.encode("utf-8"))
    return int(hash_obj.hexdigest(), 16) % num_partitions

    

def user_tier_partition(user_data: Dict, num_partitions: int) -> int:
    """
    Tier-based partitioning to handle power user skew.

    Implements custom partitioning based on user tier
    """
    # grep -r "user_tier" social_media_data/ | cut -d'"' -f16 | sort|uniq -c
    user_tier = str(user_data["user_tier"])
    user_id = str(user_data["user_id"])
    # 2. Distribute power users across multiple partitions
    if user_tier == "power_user": 
        return hash(user_id) % num_partitions
    # 3. Give other users dedicated partitions
    elif user_tier == "active_user": # second quarter)
        partition_range = max(1, num_partitions//4) # 25% of partitions
        return (hash(user_id) % partition_range) + (num_partitions // 4) 
    elif user_tier == "regular_user": # second half
        partition_range = max(1, num_partitions//4) # 25% of partitions
        return (hash(user_id) % partition_range) + (num_partitions // 2) 
    else:  # last quarter
        partition_range = max(1, num_partitions//4) # 25% of partitions
        return (hash(user_id) % partition_range) + (3 * num_partitions // 4) 



def salted_partition(user_data: Dict, num_partitions: int, salt_factor: int = 3) -> int:
    """
    Salted partitioning for hot key handling.

    """
    user_tier = user_data["user_tier"]
    user_id = str(user_data["user_id"])
    # 1. Detect if this is likely a hot key (power user)
    if user_tier == "power_user": 
        # 2. For hot keys, add random salt to distribute across partitions
        salt_id = str(random.randint(0, salt_factor-1))
        return (hash(user_id + salt_id) % num_partitions) 
    else: 
        # 3. For regular keys, use normal hash partitioning
        hash_obj = hashlib.md5(user_id.encode("utf-8"))
        return int(hash_obj.hexdigest(), 16) % num_partitions


def measure_partition_balance(partition_results: Dict[int, Dict[str, int]]) -> Dict[str, float]:
    """
    Calculate metrics to measure how well-balanced the partitions are.
    """
    # 1. Load per partition (total records) # number of entries assigned to each partition
    def accumulate_records(total_records: int, user_count_tuple: Tuple[str, int]) -> int: 
        _, count = user_count_tuple
        total_records += count 
        return total_records

    total_records = 0
    per_partition_load = defaultdict(int)
    for partition_id, partition_data in partition_results.items(): 
        partition_data_tuples = partition_data.items() # list[Dict[str, int]] --> use itertools.chain 
        records_per_partition= reduce(accumulate_records, partition_data_tuples, 0)
        
        per_partition_load[partition_id] = records_per_partition
        total_records += records_per_partition

    # 2. Balance ratio (max_load / avg_load)
    all_loads = list(per_partition_load.values())
    avg_load = sum(all_loads)/len(all_loads)
    balance_ratio = max(all_loads)/avg_load

    # 3. Standard deviation of loads
    squared_diff = [(x - avg_load)**2 for x in all_loads]
    mean_squared_diff = sum(squared_diff)/len(all_loads)
    std_dev = math.sqrt(mean_squared_diff)

    # 4. Gini coefficient for inequality measurement
    # Gini = (2 * Σ(i * xi)) / (n * Σ(xi)) - (n + 1) / n
    sorted_loads = sorted(all_loads)
    n = len(sorted_loads)
    weighted_load = sum([(idx+1) * sorted_loads[idx] for idx in range(n)])
    gini_score = (2 * weighted_load) / (n * sum(sorted_loads)) - (n+1)/n
    return {
        "total_records": total_records, 
        "balance_ratio": balance_ratio, 
        "std_dev": std_dev,
        "gini_score": gini_score
    }


def run_partitioning_experiment():
    """
    Run experiments comparing different partitioning strategies.

    """
    print("🧪 PARTITIONING EXPERIMENT: User Influence Score")
    print("="*60)

    # Define partitioning strategies to test
    strategies = [
        PartitioningStrategy("hash", hash_partition, 8),
        PartitioningStrategy("user_tier", user_tier_partition, 8),
        PartitioningStrategy("salted", salted_partition, 8),
    ]

    data_files = list(Path("social_media_data").glob("user_activity_*.jsonl"))
    if not data_files:
        print("❌ No data files found. Run data_generator.py first.")
        return

    results = {}

    for strategy in strategies:
        print(f"\n🔄 Testing {strategy.name} partitioning...")

        start_time = time.time()

        # Implement the processing logic
        mapreduce_class = UserInfluenceMapReduce
        # 1. Process files with the partitioning strategy
        per_partition_data = defaultdict(list)
        for file_path in data_files: 
            with open(file_path, "r") as f: 
                lines = f.readlines()
                for line in lines: 
                    # Tuple[int, Tuple[str, int]]
                    per_line_data = mapreduce_class.partition_aware_map(line, partitioning_strategy=strategy)
                    # int -> list[tuple[str, int]]
                    for partition_id, user_data in per_line_data:
                        per_partition_data[partition_id].append(user_data) # <-- Implicit shuffle
        
        reduced_data = mapreduce_class.partition_aware_reduce(per_partition_data)

        # 2. Measure load balance
        partition_balance = measure_partition_balance(reduced_data)

        # 3. Calculate performance metrics
        processing_time = time.time() - start_time

        # TODO: Store results for comparison
        results[strategy.name] = {
            'processing_time': processing_time,
            **partition_balance
        }

    # TODO: Print comparison results
    print("\n📊 EXPERIMENT RESULTS")
    print("-" * 40)
    # Compare strategies and identify the best one for handling skew
    print("Strategy | Processing Time | Balance Ratio | Std DEv | Gini Score")
    print("-"*40)
    for strategy, metrics in results.items():
        print(f"{strategy} | {metrics['processing_time']:.4f} | {metrics['balance_ratio']:.4f} | {metrics['std_dev']:.4f} | {metrics['gini_score']:.4f}")


def visualize_user_distribution():
    """
    Create visualizations showing the user activity distribution.

    Implement visualization of data skew
    """
    # 1. Read user activity data
    data_files = list(Path("social_media_data").glob("user_activity_*.jsonl"))
    mapreduce_class = UserInfluenceMapReduce
    per_file_data = []
    for file in data_files: 
        with open(file, "r") as f: 
            lines = f.readlines()
            for line in lines: 
                per_line_data = mapreduce_class.map(line)
                for user_data in per_line_data:
                    per_file_data.append(user_data)

    # 2. Count activities per user
    per_user_activity = mapreduce_class.reduce(per_file_data)
    user_activities = list(per_user_activity.values())
    
    
    # 3. Create plots showing:
    import matplotlib.pyplot as plt

    # User activity distribution (histogram)
    plt.figure(figsize=(12, 6))
    plt.hist(user_activities, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title("User Activity Distribution", fontsize=16, fontweight='bold')
    plt.xlabel("Number of Activities per User", fontsize=12)
    plt.ylabel("Number of Users", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Pareto chart (top users vs others)
    plt.figure(figsize=(12, 6))
    sorted_activities = sorted(user_activities, reverse=True)
    plt.scatter(range(len(sorted_activities)), sorted_activities,
                marker='o', alpha=0.6, s=30, color='red')
    plt.title("Pareto Chart: User Activity Rankings (Power User Distribution)",
              fontsize=16, fontweight='bold')
    plt.xlabel("User Rank (1 = Most Active)", fontsize=12)
    plt.ylabel("Activity Count", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale to better show skew
    plt.tight_layout()
    plt.show()

    #    - Partition load distribution for each strategy

if __name__ == "__main__":
    print("🎯 CHALLENGE 1: User Influence Score with Partitioning")
    print("="*60)
    print()


    # Uncomment when ready to test
    run_partitioning_experiment()
    visualize_user_distribution()
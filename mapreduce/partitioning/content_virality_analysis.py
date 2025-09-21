"""
Challenge 2: Content Virality Index with Partitioning Strategies

This challenge demonstrates how to approach partitioning and skew handling
for viral content analysis. Your task is to implement different partitioning
strategies and measure their effectiveness.

Problem: Rank content by engagement velocity and total reach, but handle
the fact that viral content creates massive skew in specific partitions.
"""

import json
import math
import time
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Generator
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

def get_engagement_metrics(record: Dict[str, Any]) -> Tuple[int, float, int]: 
    # 2. Calculate time factor (how recent the content is)
    # Use recency of activity to generate timestamp 
    engagement_type = record["engagement_type"]
    time_factor = {
        "share": 1.0, # most recent 
        "comment": 0.8, 
        "like": 0.6, #oldest or least values 
    }.get(engagement_type, 0.5)

    
    # 3. Extract engagement count and reach
    engagement_count = int(record["value"])
    is_viral = record.get("is_viral", False)
    hash_tag_count = len(record.get("hashtags", []))
    reach = hash_tag_count + (engagement_count * 2 if is_viral else engagement_count)
    return (engagement_count, time_factor, reach)

class ContentViralityMapReduce:
    """
    MapReduce implementation for calculating content virality index
    with different partitioning strategies.

    Virality Index = (engagement_count / time_since_creation) * reach_factor
    """

    @staticmethod
    def map(line: str) -> Generator[Tuple[str, Tuple[int, float, int]], None, None]:
        """
        Map phase: Extract content_id and engagement metrics from records.

        Parse JSON line and extract content_id, engagement_count, timestamp, reach
        Emit (content_id, (engagement_count, time_factor, reach))
        """
        # 1. Parse JSON to get content engagement data
        record = json.loads(line.strip())
        content_id = str(record["content_id"])

        # 4. Yield (content_id, (engagement_count, time_factor, reach))
        yield(content_id, get_engagement_metrics(record))

    @staticmethod
    def reduce(per_line_content_data: List[Generator], use_reduce: bool = False) -> Dict[str, float]:
        """
        Reduce phase: Calculate virality index per content.

        Aggregate content engagement metrics and compute virality score
        """
        # 1. Aggregate engagement metrics per content_id
        per_content_ecount = defaultdict(float)
        per_content_time = defaultdict(list)
        per_content_reach = defaultdict(int)
        for generator in per_line_content_data: 
            for content_data in generator:
                content_id, engagement_data_tuple = content_data
                engagement_count, time_factor, reach = engagement_data_tuple
                per_content_ecount[content_id] += engagement_count
                per_content_time[content_id].append(time_factor)
                per_content_reach[content_id] += reach

        # 2. Calculate virality index: (total_engagement / avg_time) * reach_factor
        per_content_virality_index = defaultdict(float)
        for content_id in per_content_reach.keys(): 
            total_engagement = per_content_ecount[content_id]
            avg_time = sum(per_content_time[content_id])/len(per_content_time[content_id])
            reach_factor = per_content_reach[content_id]
            per_content_virality_index[content_id] = (total_engagement/avg_time) * reach_factor

        # 3. Return {content_id: virality_index, ...}
        return dict(per_content_virality_index)

    @staticmethod
    def partition_aware_map(line: str, partitioning_strategy: PartitioningStrategy) -> Generator[Tuple[int, Tuple[str, Tuple[int, float, int]]], None, None]:
        """
        Map phase that outputs partition-aware key-value pairs.

        Parse line, determine partition, emit (partition_id, (content_id, metrics))
        """
        # 1. Parse JSON to get content data
        record = json.loads(line.strip())

        # 2. Use partitioning_strategy.partition_func to determine partition
        content_id = str(record["content_id"])
        
        engagement_metrics = get_engagement_metrics(record)

        # 3. Get Parition Id 
        partition_id = partitioning_strategy.partition_func(record, partitioning_strategy.num_partitions)

        # 3. Yield (partition_id, (content_id, engagement_metrics))
        yield(partition_id, (content_id, engagement_metrics))

    @staticmethod
    def partition_aware_reduce(partitioned_data: Dict[int, List[Tuple[str, Tuple[int, float, int]]]]) -> Dict[int, Dict[str, float]]:
        """
        Reduce phase that processes each partition separately.

        Process each partition's content data and return per-partition virality results
        """
        # 1. For each partition, aggregate content metrics
        per_parition_per_content_ecount = defaultdict(lambda: defaultdict(int))
        per_parition_per_content_time = defaultdict(lambda: defaultdict(list))
        per_parition_per_content_reach = defaultdict(lambda: defaultdict(int))
        for pid, pdata in partitioned_data.items():
            for content_data in pdata: 
                content_id, engagement_metrics = content_data
                engagement_count, time_factor, reach = engagement_metrics
                per_parition_per_content_ecount[pid][content_id] += engagement_count
                per_parition_per_content_time[pid][content_id].append(time_factor)
                per_parition_per_content_reach[pid][content_id] += reach

        # 2. Calculate virality index per content in each partition
        per_parition_per_content_virality = defaultdict(lambda: defaultdict(float))
        for pid in per_parition_per_content_ecount.keys(): 
            for content_id in per_parition_per_content_ecount[pid].keys():
                total_engagement = per_parition_per_content_ecount[pid][content_id]
                avg_time = sum(per_parition_per_content_time[pid][content_id])/len(per_parition_per_content_time[pid][content_id])
                reach_factor = per_parition_per_content_reach[pid][content_id]
                per_parition_per_content_virality[pid][content_id] = (total_engagement/avg_time) * reach_factor

        # 3. Return {partition_id: {content_id: virality_index, ...}, ...}
        return {pid: dict(content_dict) for pid, content_dict in per_parition_per_content_virality.items()}


def hash_partition(content_data: Dict, num_partitions: int) -> int:
    """
    Simple hash-based partitioning.

    Implements hash partitioning strategy for content
    """
    # 1. Extract content_id from content_data
    content_id = str(content_data["content_id"])
    # 2. Use hash function to distribute content across partitions
    ho = hashlib.md5(content_id.encode("utf-8"))
    hash_id = int(ho.hexdigest(), 16)
    # 3. Return partition_id
    return hash_id % num_partitions

def engagement_tier_partition(content_data: Dict, num_partitions: int) -> int:
    """
    Tier-based partitioning to handle viral content skew.

    Implements custom partitioning based on engagement level
    """
    content_id = str(content_data["content_id"])

    # 1. Classify content by engagement level (viral, popular, normal, low)
    is_viral = content_data.get("is_viral", False)
    engagement_value = content_data.get("value", 0)
  
    # 2. Distribute viral content across multiple partitions
    if is_viral: 
        return hash(content_id) % num_partitions
    
    # 3. Give other tiers dedicated partition ranges
    elif engagement_value >= 100: 
        partition_range = max(1, num_partitions // 4)
        return hash(content_id) % partition_range 
    
    elif engagement_value >=50: 
        partition_range = max(1, num_partitions// 4)
        return num_partitions // 4 + hash(content_id) % partition_range 
    
    else: 
        partition_range = max(1, num_partitions//4)
        return 3 * num_partitions // 4 + hash(content_id) % partition_range


def temporal_partition(content_data: Dict, num_partitions: int) -> int:
    """
    Time-based partitioning for temporal analysis.

    Partitions content by creation time to handle temporal skew
    """
    content_id = content_data["content_id"]
    # 1. Extract timestamp from content_data
    engagement_type = content_data["engagement_type"]
    time_factor = {
        "share": 1.0, # most recent 
        "comment": 0.8, 
        "like": 0.6, #oldest or least values 
    }.get(engagement_type, 0.5)

    # 2. Create time-based buckets (e.g., hourly, daily)
    # 3. Distribute across partitions based on time bucket
    # 4. Return partition_id based on temporal bucket
    if time_factor >= 0.9: 
        # distribute evenly
        return hash(content_id) % num_partitions
    elif time_factor >=0.6: 
        partition_range = max(1, num_partitions //4)
        return num_partitions//2 + hash(content_id) % partition_range
    else: 
        partition_range = max(1, num_partitions//4)
        return 3*num_partitions//4 + hash(content_id) % partition_range


def hybrid_partition(content_data: Dict, num_partitions: int) -> int:
    """
    Hybrid partitioning combining engagement and temporal factors.

    Advanced partitioning that considers both virality and time
    """
    content_id = str(content_data["content_id"])

    # 1. Get engagement metrics
    is_viral = content_data.get("is_viral", False)
    engagement_value = content_data.get("value", 0)

    # 2. Get temporal metrics
    engagement_type = content_data["engagement_type"]
    time_factor = {
        "share": 1.0,    # most recent 
        "comment": 0.8,  # medium age
        "like": 0.6,     # oldest
    }.get(engagement_type, 0.5)

    # 3. Apply different strategies for viral vs normal content
    if is_viral:
        # Viral content: prioritize time-based distribution to handle spikes
        if time_factor >= 0.9:  # Recent viral content (hot!)
            # Spread recent viral across ALL partitions to prevent hotspots
            return hash(content_id) % num_partitions
        else:  # Older viral content
            # Less critical, can cluster in dedicated partitions
            partition_range = max(1, num_partitions // 4)
            return hash(content_id) % partition_range

    # 4. For non-viral content: engagement-first, then temporal
    elif engagement_value >= 100:  # High engagement, non-viral
        # High engagement: group by time for efficient temporal queries
        if time_factor >= 0.8:  # Recent high-engagement
            partition_range = max(1, num_partitions // 8)
            return (num_partitions // 4) + (hash(content_id) % partition_range)
        else:  # Older high-engagement
            partition_range = max(1, num_partitions // 8)
            return (3 * num_partitions // 8) + (hash(content_id) % partition_range)

    elif engagement_value >= 50:  # Medium engagement
        # Medium engagement: balanced distribution
        partition_range = max(1, num_partitions // 4)
        return (num_partitions // 2) + (hash(content_id) % partition_range)

    else:  # Low engagement
        # Low engagement: simple time-based clustering
        if time_factor >= 0.8:  # Recent low-engagement
            partition_range = max(1, num_partitions // 8)
            return (3 * num_partitions // 4) + (hash(content_id) % partition_range)
        else:  # Old low-engagement
            partition_range = max(1, num_partitions // 8)
            return (7 * num_partitions // 8) + (hash(content_id) % partition_range)


def measure_virality_balance(partition_results: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    """
    Calculate metrics to measure how well-balanced the partitions are for virality analysis.
    """
    # 1. Calculate load per partition (number of content items)
    # 2. Calculate virality concentration (sum of virality scores per partition)
    def acc_virality(total_virality: float, content_virality_tuple: Tuple[str, float]) -> float: 
        content_id, virality = content_virality_tuple 
        total_virality+=virality
        return total_virality 
    
    per_partition_load = defaultdict(int)
    per_partition_virality = defaultdict(float)
    for pid, pdata in partition_results.items():
        pdata_tuples = pdata.items() # per content data 
        total_virality_per_pid = reduce(acc_virality, pdata_tuples, 0.0)
        per_partition_virality[pid] = total_virality_per_pid 
        per_partition_load[pid] = len(pdata_tuples)


    
    # 3. Measure both count balance and value balance 
    # max_load / avg_load 
    all_load = list(per_partition_load.values())
    avg_load = sum(all_load)/len(all_load)
    balance_load = max(all_load)/avg_load

    all_virality = list(per_partition_virality.values())
    avg_virality = sum(all_virality)/len(all_virality)
    balance_virality = max(all_virality)/avg_virality

    # 4. Return metrics including balance_ratio, std_deviation, gini_coefficient
    # 3. Standard deviation of loads
    squared_diff = [(x - avg_load)**2 for x in all_load]
    mean_squared_diff = sum(squared_diff)/len(all_load)
    std_dev = math.sqrt(mean_squared_diff)

    # 4. Gini coefficient for inequality measurement
    # Gini = (2 * Σ(i * xi)) / (n * Σ(xi)) - (n + 1) / n
    sorted_loads = sorted(all_load)
    n = len(sorted_loads)
    weighted_load = sum([(idx+1) * sorted_loads[idx] for idx in range(n)])
    gini_score = (2 * weighted_load) / (n * sum(sorted_loads)) - (n+1)/n
    
    # 5. Add virality-specific metrics like viral_content_distribution
    viral_partitions = sum(1 for v in all_virality if v >= avg_virality)
    viral_concentration = viral_partitions / len(all_virality)

    return {
        "balance_load": balance_load,
        "balance_virality": balance_virality,
        "std_dev": std_dev,
        "gini_score": gini_score,
        "viral_concentration": viral_concentration
    }

def run_virality_experiment():
    """
    Run experiments comparing different partitioning strategies for content virality.
    """
    print("🧪 PARTITIONING EXPERIMENT: Content Virality Index")
    print("="*60)

    # Define partitioning strategies to test
    strategies = [
        PartitioningStrategy("hash", hash_partition, 8),
        PartitioningStrategy("engagement_tier", engagement_tier_partition, 8),
        PartitioningStrategy("temporal", temporal_partition, 8),
        PartitioningStrategy("hybrid", hybrid_partition, 8),
    ]

    # Look for content engagement data files
    data_files = list(Path("social_media_data").glob("content_engagement_*.jsonl"))
    if not data_files:
        print("❌ No content engagement data files found. Run data_generator.py first.")
        return

    results = {}

    for strategy in strategies:
        print(f"\n🔄 Testing {strategy.name} partitioning...")

        start_time = time.time()
        mapreduce_class = ContentViralityMapReduce
        # 1. Process files with the partitioning strategy
        per_partition_data = defaultdict(list)
        for file_path in data_files: 
            with open(file_path, "r") as f:
                lines = f.readlines()
                for line in lines: 
                    per_line_data = mapreduce_class.partition_aware_map(line, strategy) # generator 
                    for partition_id, content_data in per_line_data: 
                        per_partition_data[partition_id].append(content_data)
        
        reduced_data = mapreduce_class.partition_aware_reduce(per_partition_data)
                    
        # 2. Measure load balance and virality distribution
        virality_balance = measure_virality_balance(reduced_data)

        # 3. Calculate performance metrics
        processing_time = time.time() - start_time

        # 4. Store results for comparison
        results[strategy.name] = {
            'processing_time': processing_time,
            **virality_balance
        }

    # TODO: Print comparison results
    print("\n📊 EXPERIMENT RESULTS")
    print("-" * 40)
    print("Strategy | Processing Time | Balance Load Ratio | Balance Virality Ratio | Viral Concentration | Std Dev | Gini Score")
    print("-"*80)
    for strategy, metrics in results.items():
        print(f"{strategy} | {metrics['processing_time']:.4f} | {metrics['balance_load']:.4f} | {metrics['balance_virality']:.4f} |{metrics['viral_concentration']:.4f} | {metrics['std_dev']:.4f} | {metrics['gini_score']:.4f}")



def visualize_virality_distribution():
    """
    Create visualizations showing the content virality distribution.

    Implement visualization of virality skew
    """
    import matplotlib.pyplot as plt

    # 1. Read content engagement data
    data_files = list(Path("social_media_data").glob("content_engagement_*.jsonl"))
    mapreduce_class = ContentViralityMapReduce
    per_file_data = []

    for file in data_files:
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                per_line_data = mapreduce_class.map(line)
                for content_data in per_line_data:
                    per_file_data.append(content_data)

    # 2. Calculate virality scores per content
    per_content_virality = mapreduce_class.reduce([iter([data]) for data in per_file_data])
    virality_scores = list(per_content_virality.values())

    # 3. Create plots showing:

    # Virality distribution (histogram with log scale)
    plt.figure(figsize=(12, 6))
    plt.hist(virality_scores, bins=50, alpha=0.7, color='orange', edgecolor='black')
    plt.title("Content Virality Distribution", fontsize=16, fontweight='bold')
    plt.xlabel("Virality Index Score", fontsize=12)
    plt.ylabel("Number of Content Items", fontsize=12)
    plt.yscale('log')  # Log scale to better show skew
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Top viral content (Pareto chart)
    plt.figure(figsize=(12, 6))
    sorted_virality = sorted(virality_scores, reverse=True)
    plt.scatter(range(len(sorted_virality)), sorted_virality,
                marker='o', alpha=0.6, s=30, color='purple')
    plt.title("Pareto Chart: Content Virality Rankings (Viral Content Distribution)",
              fontsize=16, fontweight='bold')
    plt.xlabel("Content Rank (1 = Most Viral)", fontsize=12)
    plt.ylabel("Virality Index Score", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale to better show viral content dominance
    plt.tight_layout()
    plt.show()

    # Temporal virality patterns by engagement type
    plt.figure(figsize=(12, 6))

    # Collect virality by engagement type
    virality_by_type = {"share": [], "comment": [], "like": [], "save": [], "view": []}

    for file in data_files:
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                record = json.loads(line.strip())
                content_id = str(record["content_id"])
                engagement_type = record["engagement_type"]

                if content_id in per_content_virality:
                    virality_by_type[engagement_type].append(per_content_virality[content_id])

    # Create box plot showing virality distribution by engagement type
    plt.boxplot([virality_by_type["share"], virality_by_type["comment"], virality_by_type["like"], virality_by_type["save"], virality_by_type["view"]],
                labels=["Share (Recent)", "Comment (Medium)", "Like (Older)", "Save", "View"])
    plt.title("Virality Patterns by Engagement Type (Temporal Analysis)",
              fontsize=16, fontweight='bold')
    plt.xlabel("Engagement Type (Recency Proxy)", fontsize=12)
    plt.ylabel("Virality Index Score", fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Partition load distribution for each strategy
    plt.figure(figsize=(15, 10))

    strategies = [
        PartitioningStrategy("hash", hash_partition, 8),
        PartitioningStrategy("engagement_tier", engagement_tier_partition, 8),
        PartitioningStrategy("temporal", temporal_partition, 8),
        PartitioningStrategy("hybrid", hybrid_partition, 8),
    ]

    for i, strategy in enumerate(strategies, 1):
        plt.subplot(2, 2, i)

        # Process data with this strategy
        per_partition_data = defaultdict(list)
        for file_path in data_files:
            with open(file_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    per_line_data = mapreduce_class.partition_aware_map(line, strategy)
                    for partition_id, content_data in per_line_data:
                        per_partition_data[partition_id].append(content_data)

        reduced_data = mapreduce_class.partition_aware_reduce(per_partition_data)

        # Calculate virality per partition
        partition_virality = {}
        for pid, content_dict in reduced_data.items():
            partition_virality[pid] = sum(content_dict.values())

        # Plot partition loads
        partitions = list(range(max(partition_virality.keys()) + 1))
        loads = [partition_virality.get(p, 0) for p in partitions]

        plt.bar(partitions, loads, alpha=0.7)
        plt.title(f"{strategy.name.title()} Partitioning", fontsize=12, fontweight='bold')
        plt.xlabel("Partition ID", fontsize=10)
        plt.ylabel("Total Virality Score", fontsize=10)
        plt.grid(True, alpha=0.3)

    plt.suptitle("Partition Load Distribution Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print(f"\n📊 Virality Analysis Summary:")
    print(f"Total content items analyzed: {len(virality_scores)}")
    print(f"Average virality score: {sum(virality_scores)/len(virality_scores):.2f}")
    print(f"Max virality score: {max(virality_scores):.2f}")
    print(f"Viral content skew ratio: {max(virality_scores)/sum(virality_scores)*len(virality_scores):.2f}x")


def analyze_viral_hotspots():
    """
    Identify and analyze viral content hotspots that cause partitioning issues.

    Implement viral hotspot analysis
    """
    print("🔥 VIRAL HOTSPOT ANALYSIS")
    print("="*50)

    # Read content engagement data
    data_files = list(Path("social_media_data").glob("content_engagement_*.jsonl"))
    mapreduce_class = ContentViralityMapReduce
    per_file_data = []

    for file in data_files:
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                per_line_data = mapreduce_class.map(line)
                for content_data in per_line_data:
                    per_file_data.append(content_data)

    # Calculate virality scores per content
    per_content_virality = mapreduce_class.reduce([iter([data]) for data in per_file_data])

    # 1. Identify content that exceeds virality thresholds
    virality_scores = list(per_content_virality.values())
    avg_virality = sum(virality_scores) / len(virality_scores)

    # Define hotspot thresholds
    high_threshold = avg_virality * 5    # 5x average = hot
    extreme_threshold = avg_virality * 10  # 10x average = extreme hotspot

    hot_content = {cid: score for cid, score in per_content_virality.items()
                   if score >= high_threshold}
    extreme_content = {cid: score for cid, score in per_content_virality.items()
                       if score >= extreme_threshold}

    print(f"\n📊 Hotspot Identification:")
    print(f"Average virality score: {avg_virality:.2f}")
    print(f"Hot content threshold (5x avg): {high_threshold:.2f}")
    print(f"Extreme threshold (10x avg): {extreme_threshold:.2f}")
    print(f"Hot content items: {len(hot_content)} ({len(hot_content)/len(per_content_virality)*100:.1f}%)")
    print(f"Extreme content items: {len(extreme_content)} ({len(extreme_content)/len(per_content_virality)*100:.1f}%)")

    # 2. Analyze temporal clustering of viral events
    print(f"\n⏰ Temporal Clustering Analysis:")

    viral_by_type = {"share": [], "comment": [], "like": [], "save": [], "view": []}
    hot_by_type = {"share": 0, "comment": 0, "like": 0, "save": 0, "view": 0}

    for file in data_files:
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                record = json.loads(line.strip())
                content_id = str(record["content_id"])
                engagement_type = record["engagement_type"]

                if content_id in per_content_virality:
                    virality = per_content_virality[content_id]
                    viral_by_type[engagement_type].append(virality)

                    if virality >= high_threshold:
                        hot_by_type[engagement_type] += 1

    for eng_type in viral_by_type:
        if viral_by_type[eng_type]:
            avg_by_type = sum(viral_by_type[eng_type]) / len(viral_by_type[eng_type])
            print(f"{eng_type.capitalize()} events - Avg virality: {avg_by_type:.2f}, Hot items: {hot_by_type[eng_type]}")

    # 3. Measure impact on partition balance
    print(f"\n⚖️ Partition Balance Impact Analysis:")

    strategies = [
        PartitioningStrategy("hash", hash_partition, 8),
        PartitioningStrategy("engagement_tier", engagement_tier_partition, 8),
        PartitioningStrategy("temporal", temporal_partition, 8),
        PartitioningStrategy("hybrid", hybrid_partition, 8),
    ]

    balance_impact = {}

    for strategy in strategies:
        # Process data with this strategy
        per_partition_data = defaultdict(list)
        for file_path in data_files:
            with open(file_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    per_line_data = mapreduce_class.partition_aware_map(line, strategy)
                    for partition_id, content_data in per_line_data:
                        per_partition_data[partition_id].append(content_data)

        reduced_data = mapreduce_class.partition_aware_reduce(per_partition_data)

        # Find which partitions contain hot content
        hot_partitions = set()
        hotspot_concentration = {}

        for pid, content_dict in reduced_data.items():
            partition_hot_count = 0
            partition_hot_virality = 0

            for cid, virality in content_dict.items():
                if virality >= high_threshold:
                    hot_partitions.add(pid)
                    partition_hot_count += 1
                    partition_hot_virality += virality

            if partition_hot_count > 0:
                hotspot_concentration[pid] = {
                    'hot_count': partition_hot_count,
                    'hot_virality': partition_hot_virality,
                    'total_items': len(content_dict)
                }

        # Calculate balance metrics for this strategy
        total_partitions = len(reduced_data)
        hotspot_ratio = len(hot_partitions) / total_partitions if total_partitions > 0 else 0

        if hotspot_concentration:
            max_hot_count = max(data['hot_count'] for data in hotspot_concentration.values())
            avg_hot_count = sum(data['hot_count'] for data in hotspot_concentration.values()) / len(hotspot_concentration)
            hot_imbalance = max_hot_count / avg_hot_count if avg_hot_count > 0 else 0
        else:
            hot_imbalance = 0

        balance_impact[strategy.name] = {
            'hotspot_ratio': hotspot_ratio,
            'hot_imbalance': hot_imbalance,
            'affected_partitions': len(hot_partitions)
        }

        print(f"{strategy.name}: {len(hot_partitions)}/{total_partitions} partitions affected "
              f"({hotspot_ratio*100:.1f}%), imbalance ratio: {hot_imbalance:.2f}")

    # 4. Suggest mitigation strategies
    print(f"\n💡 Mitigation Strategy Recommendations:")

    # Find best performing strategy
    best_strategy = min(balance_impact.keys(),
                       key=lambda x: balance_impact[x]['hot_imbalance'])
    worst_strategy = max(balance_impact.keys(),
                        key=lambda x: balance_impact[x]['hot_imbalance'])

    print(f"✅ Best performing strategy: {best_strategy}")
    print(f"   - Lowest hotspot imbalance: {balance_impact[best_strategy]['hot_imbalance']:.2f}")
    print(f"   - Affected partitions: {balance_impact[best_strategy]['affected_partitions']}")

    print(f"❌ Worst performing strategy: {worst_strategy}")
    print(f"   - Highest hotspot imbalance: {balance_impact[worst_strategy]['hot_imbalance']:.2f}")
    print(f"   - Affected partitions: {balance_impact[worst_strategy]['affected_partitions']}")

    print(f"\n🎯 Recommended Actions:")

    if len(extreme_content) > 0:
        print(f"1. CRITICAL: {len(extreme_content)} extreme viral items detected!")
        print(f"   - Consider dedicated processing for items >10x average virality")
        print(f"   - Implement real-time hotspot detection and redistribution")

    if balance_impact[worst_strategy]['hot_imbalance'] > 3.0:
        print(f"2. HIGH: {worst_strategy} partitioning shows severe imbalance (>3x)")
        print(f"   - Avoid {worst_strategy} for viral workloads")
        print(f"   - Implement dynamic load balancing")

    if hot_by_type['share'] > hot_by_type['like'] * 2:
        print(f"3. TEMPORAL: Share events show higher viral concentration")
        print(f"   - Consider time-decay factors in partitioning")
        print(f"   - Implement sliding window for recent viral content")

    print(f"4. GENERAL: Use {best_strategy} partitioning for better viral content distribution")
    print(f"5. MONITORING: Set up alerts for content exceeding {high_threshold:.1f} virality score")

    return {
        'hot_content_count': len(hot_content),
        'extreme_content_count': len(extreme_content),
        'best_strategy': best_strategy,
        'worst_strategy': worst_strategy,
        'balance_impact': balance_impact
    }


if __name__ == "__main__":
    print("🎯 CHALLENGE 2: Content Virality Index with Partitioning")
    print("="*60)
    print()
    
    # Uncomment when ready to test
    run_virality_experiment()
    visualize_virality_distribution()
    analyze_viral_hotspots()
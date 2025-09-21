"""
Challenge 3: Hashtag Trending Analysis with Partitioning Strategies

This challenge demonstrates how to approach partitioning and skew handling
for hashtag trending analysis with co-occurrence computation. Your task is to
implement different partitioning strategies and measure their effectiveness.

Problem: Real-time hashtag frequency with co-occurrence analysis, but handle
the fact that popular hashtags dominate certain partitions and require data
locality for efficient co-occurrence computation.
"""

import json
import math
import time
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Generator, Set
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


class HashtagTrendingMapReduce:
    """
    MapReduce implementation for hashtag trending analysis with co-occurrence
    computation using different partitioning strategies.

    Trending Score = (frequency * recency_factor) + co_occurrence_boost
    """

    @staticmethod
    def map(line: str) -> Generator[Tuple[str, Tuple[int, float, Set[str]]], None, None]:
        """
        Map phase: Extract hashtags and co-occurrence data from records.

        Parse JSON line and extract hashtags, calculate recency factor, find co-occurring tags
        Emit (hashtag, (frequency_count, recency_factor, co_occurring_tags))
        """
        # 1. Parse JSON to get content engagement data with hashtags
        record = json.loads(line.strip())
        
        # 2. Calculate recency factor based on engagement type
        engagement_type = record["engagement_type"]
        recency_factor = {
            "share": 1.0, # most recent 
            "comment": 0.8, 
            "like": 0.6, #oldest or least values 
        }.get(engagement_type, 0.5)

        # 3. Extract all hashtags from the content
        hash_tags = list(record["hashtags"])
        for hashtag in hash_tags:
            other_hashtags_in_post = set(hash_tags) - {hashtag}
            yield(hashtag, (1, recency_factor, other_hashtags_in_post))
        
    @staticmethod
    def reduce(per_line_hashtag_data: List[Generator], use_reduce: bool = False) -> Dict[str, Dict[str, any]]:
        """
        Reduce phase: Calculate trending scores and co-occurrence matrices per hashtag.

        Aggregate hashtag metrics and compute trending scores with co-occurrence analysis
        """
        # 1. Aggregate frequency counts per hashtag
        per_hashtag_count = defaultdict(int)
        per_hashtag_recency_factor = defaultdict(list)
        hashtag_cooccurence_matrix = defaultdict(lambda: defaultdict(int))
        for gen in per_line_hashtag_data: 
            for hashtag, ht_metrics in gen: 
                count, recency_factor, other_hashtags_in_post = ht_metrics
                per_hashtag_count[hashtag] += count 
                per_hashtag_recency_factor[hashtag].append(recency_factor)

                # 3. Build co-occurrence matrix (which hashtags appear together)
                for ot_hashtag in other_hashtags_in_post:
                    hashtag_cooccurence_matrix[hashtag][ot_hashtag] += 1
                    hashtag_cooccurence_matrix[ot_hashtag][hashtag] += 1

                

        # 2. Calculate average recency factor per hashtag
        per_hashtag_avg_recency = defaultdict(float)
        for hashtag, recency_factor_list in per_hashtag_recency_factor.items():
            per_hashtag_avg_recency[hashtag] = sum(recency_factor_list)/len(recency_factor_list)

        # 4. Calculate trending score: (frequency * avg_recency) + co_occurrence_boost
        per_hashtag_trending_score = defaultdict(float)
        for hashtag in per_hashtag_count.keys(): 
            f = per_hashtag_count[hashtag]
            ar = per_hashtag_avg_recency[hashtag]
            cb = sum(list(hashtag_cooccurence_matrix[hashtag].values()))
            per_hashtag_trending_score[hashtag] = (f * ar) + cb
        # 5. Return {hashtag: {'trending_score': score, 'frequency': freq, 'co_occurrences': dict}, ...}
        per_hashtag_metrics = {}
        for hashtag in per_hashtag_count.keys(): 
            metrics = {
                "trending_score": per_hashtag_trending_score[hashtag], 
                "frequency": per_hashtag_count[hashtag], 
                "co_occurrences": dict(hashtag_cooccurence_matrix[hashtag]), 
            }
            per_hashtag_metrics[hashtag] = metrics 
        return dict(per_hashtag_metrics)


    @staticmethod
    def partition_aware_map(line: str, partitioning_strategy: PartitioningStrategy) -> Generator[Tuple[int, Tuple[str, Tuple[int, float, Set[str]]]], None, None]:
        """
        Map phase that outputs partition-aware key-value pairs.

        Parse line, determine partition, emit (partition_id, (hashtag, metrics))
        """
        # 1. Parse JSON to get hashtag data
        record = json.loads(line.strip())

        # 2. Calculate recency factor based on engagement type
        engagement_type = record["engagement_type"]
        recency_factor = {
            "share": 1.0, # most recent
            "comment": 0.8,
            "like": 0.6, #oldest or least values
        }.get(engagement_type, 0.5)

        # 3. Extract all hashtags from the content
        hash_tags = list(record["hashtags"])

        for hashtag in hash_tags:
            other_hashtags_in_post = set(hash_tags) - {hashtag}

            # 4. Use partitioning_strategy.partition_func to determine partition
            # Create hashtag_data dict for the partitioning function
            hashtag_data = {
                "hashtag": hashtag,
                "record": record,  # Include full record for context
                "hashtags": hash_tags,
                "other_hashtags": other_hashtags_in_post,
                "recency_factor": recency_factor
            }

            partition_id = partitioning_strategy.partition_func(hashtag_data, partitioning_strategy.num_partitions)

            # 5. Yield (partition_id, (hashtag, hashtag_metrics))
            yield (partition_id, (hashtag, (1, recency_factor, other_hashtags_in_post)))

    @staticmethod
    def partition_aware_reduce(partitioned_data: Dict[int, List[Tuple[str, Tuple[int, float, Set[str]]]]]) -> Dict[int, Dict[str, Dict[str, any]]]:
        """
        Reduce phase that processes each partition separately.

        Process each partition's hashtag data and return per-partition trending results
        """
        result = {}

        # 1. For each partition, aggregate hashtag metrics
        for partition_id, hashtag_tuples in partitioned_data.items():
            # Initialize per-partition data structures
            per_hashtag_count = defaultdict(int)
            per_hashtag_recency_factor = defaultdict(list)
            hashtag_cooccurence_matrix = defaultdict(lambda: defaultdict(int))

            # Process each hashtag in this partition
            for hashtag, ht_metrics in hashtag_tuples:
                count, recency_factor, other_hashtags_in_post = ht_metrics
                per_hashtag_count[hashtag] += count
                per_hashtag_recency_factor[hashtag].append(recency_factor)

                # 3. Build co-occurrence matrices per partition
                for ot_hashtag in other_hashtags_in_post:
                    hashtag_cooccurence_matrix[hashtag][ot_hashtag] += 1
                    hashtag_cooccurence_matrix[ot_hashtag][hashtag] += 1

            # 2. Calculate trending scores within each partition
            per_hashtag_avg_recency = {}
            for hashtag, recency_factor_list in per_hashtag_recency_factor.items():
                per_hashtag_avg_recency[hashtag] = sum(recency_factor_list) / len(recency_factor_list)

            # Calculate trending scores for this partition
            per_hashtag_trending_score = {}
            for hashtag in per_hashtag_count.keys():
                f = per_hashtag_count[hashtag]
                ar = per_hashtag_avg_recency[hashtag]
                cb = sum(hashtag_cooccurence_matrix[hashtag].values())
                per_hashtag_trending_score[hashtag] = (f * ar) + cb

            # Build final metrics for this partition
            per_hashtag_metrics = {}
            for hashtag in per_hashtag_count.keys():
                metrics = {
                    "trending_score": per_hashtag_trending_score[hashtag],
                    "frequency": per_hashtag_count[hashtag],
                    "co_occurrences": dict(hashtag_cooccurence_matrix[hashtag]),
                    "avg_recency": per_hashtag_avg_recency[hashtag]
                }
                per_hashtag_metrics[hashtag] = metrics

            # 4. Store results for this partition
            result[partition_id] = per_hashtag_metrics

        return result


def hash_partition(hashtag_data: Dict, num_partitions: int) -> int:
    """
    Simple hash-based partitioning for hashtags.

    Implements hash partitioning strategy for hashtag distribution
    """
    # hashtag_data = {
    #     "hashtag": hashtag,
    #     "record": record,  # Include full record for context
    #     "hashtags": hash_tags,
    #     "other_hashtags": other_hashtags_in_post,
    #     "recency_factor": recency_factor
    # }

    # 1. Extract primary hashtag from hashtag_data
    hashtag = hashtag_data["hashtag"]

    # 2. Use hash function to distribute hashtags across partitions
    ho = hashlib.md5(str(hashtag).encode("utf-8"))

    # 3. Return partition_id
    return int(ho.hexdigest(), 16) % num_partitions


def popularity_tier_partition(hashtag_data: Dict, num_partitions: int) -> int:
    """
    Tier-based partitioning to handle popular hashtag skew.

    Implements custom partitioning based on hashtag popularity/frequency
    """
    hashtag = hashtag_data["hashtag"]
    record = hashtag_data["record"]
    other_hashtags = hashtag_data["other_hashtags"]

    # 1. Classify hashtags by popularity level (trending, popular, normal, niche)
    # Use multiple signals to determine popularity

    # Signal 1: Is this marked as viral content?
    is_viral_content = record.get("is_viral", False)

    # Signal 2: High engagement value
    engagement_value = record.get("value", 0)

    # Signal 3: Co-occurrence count (popular hashtags appear with many others)
    co_occurrence_count = len(other_hashtags)

    # Signal 4: Common hashtag patterns (starting with common prefixes)
    is_trending_pattern = hashtag.lower().startswith(('#trending', '#viral', '#breaking', '#hot'))

    # 2. Classify into tiers based on signals
    if is_viral_content or is_trending_pattern or engagement_value >= 150:
        # TRENDING: Distribute across ALL partitions to prevent hotspots
        return hash(hashtag) % num_partitions

    elif engagement_value >= 100 or co_occurrence_count >= 3:
        # POPULAR: First quarter of partitions
        partition_range = max(1, num_partitions // 4)
        return hash(hashtag) % partition_range

    elif engagement_value >= 50 or co_occurrence_count >= 1:
        # NORMAL: Second quarter of partitions
        partition_range = max(1, num_partitions // 4)
        return (hash(hashtag) % partition_range) + (num_partitions // 4)

    else:
        # NICHE: Remaining partitions (second half)
        partition_range = max(1, num_partitions // 2)
        return (hash(hashtag) % partition_range) + (num_partitions // 2)


def co_occurrence_partition(hashtag_data: Dict, num_partitions: int) -> int:
    """
    Co-occurrence aware partitioning for data locality.

    Partitions hashtags to maximize co-occurrence locality within partitions
    """
    hashtag = hashtag_data["hashtag"]
    other_hashtags = hashtag_data["other_hashtags"]
    engagement_value = hashtag_data["record"].get("value", 0)

    # 1. Extract hashtag and its co-occurring tags
    all_hashtags_in_post = [hashtag] + list(other_hashtags)

    # 2. Use consistent hashing to ensure co-occurring tags end up in same partition
    # Strategy: Hash the sorted set of all hashtags in the post
    # This ensures that all hashtags from the same post get the same "group hash"

    if len(all_hashtags_in_post) > 1:
        # For posts with multiple hashtags, use the sorted combination
        # This ensures all hashtags in the same post tend towards the same partition
        sorted_hashtags = sorted(all_hashtags_in_post)

        # Create a stable hash from the combination of hashtags
        combined_hash_string = "|".join(sorted_hashtags)
        group_hash = hashlib.md5(combined_hash_string.encode("utf-8"))
        base_partition = int(group_hash.hexdigest(), 16) % num_partitions

        # 3. Consider hashtag frequency in partition assignment
        # For high-engagement posts, add some variation to prevent hotspots
        if engagement_value >= 100:
            # High-engagement: add slight variation based on individual hashtag
            individual_hash = hash(hashtag) % 3  # 0, 1, or 2
            return (base_partition + individual_hash) % num_partitions
        else:
            # Normal engagement: keep all hashtags from post in same partition
            return base_partition

    else:
        # Single hashtag posts: use simple hash partitioning
        return hash(hashtag) % num_partitions


def semantic_cluster_partition(hashtag_data: Dict, num_partitions: int) -> int:
    """
    Semantic clustering partitioning for related hashtags.

    Advanced partitioning that groups semantically similar hashtags
    """
    hashtag = hashtag_data["hashtag"].lower()
    other_hashtags = {h.lower() for h in hashtag_data["other_hashtags"]}
    engagement_value = hashtag_data["record"].get("value", 0)

    # 1. Define semantic clusters with keyword patterns
    semantic_clusters = {
        "technology": {
            "keywords": ["ai", "ml", "tech", "data", "software", "coding", "programming", "digital", "cyber", "robot", "algorithm", "blockchain", "crypto"],
            "partition_range": (0, max(1, num_partitions // 8))  # First 12.5%
        },
        "sports": {
            "keywords": ["sport", "football", "soccer", "basketball", "baseball", "tennis", "golf", "olympics", "fifa", "nfl", "nba", "game", "match", "team"],
            "partition_range": (max(1, num_partitions // 8), max(2, num_partitions // 4))  # Next 12.5%
        },
        "entertainment": {
            "keywords": ["movie", "film", "music", "song", "artist", "celebrity", "actor", "singer", "band", "concert", "album", "tv", "show", "series"],
            "partition_range": (max(2, num_partitions // 4), max(3, 3 * num_partitions // 8))  # Next 12.5%
        },
        "politics": {
            "keywords": ["politics", "election", "vote", "government", "president", "congress", "senate", "democrat", "republican", "policy", "law", "biden", "trump"],
            "partition_range": (max(3, 3 * num_partitions // 8), max(4, num_partitions // 2))  # Next 12.5%
        },
        "business": {
            "keywords": ["business", "money", "finance", "stock", "market", "economy", "investment", "startup", "company", "ceo", "sales", "profit", "banking"],
            "partition_range": (max(4, num_partitions // 2), max(5, 5 * num_partitions // 8))  # Next 12.5%
        },
        "health": {
            "keywords": ["health", "medical", "doctor", "hospital", "medicine", "covid", "vaccine", "virus", "fitness", "wellness", "mental", "therapy"],
            "partition_range": (max(5, 5 * num_partitions // 8), max(6, 3 * num_partitions // 4))  # Next 12.5%
        },
        "lifestyle": {
            "keywords": ["food", "recipe", "cooking", "travel", "fashion", "style", "beauty", "home", "design", "art", "photo", "love", "family", "life"],
            "partition_range": (max(6, 3 * num_partitions // 4), max(7, 7 * num_partitions // 8))  # Next 12.5%
        },
        "news": {
            "keywords": ["news", "breaking", "update", "alert", "report", "story", "event", "happening", "latest", "urgent", "live", "press"],
            "partition_range": (max(7, 7 * num_partitions // 8), num_partitions)  # Last 12.5%
        }
    }

    # 2. Analyze hashtag for semantic similarity
    def get_semantic_cluster(tag):
        tag_clean = tag.replace("#", "").lower()

        # Check for exact keyword matches
        for cluster_name, cluster_info in semantic_clusters.items():
            for keyword in cluster_info["keywords"]:
                if keyword in tag_clean:
                    return cluster_name

        # Check co-occurring hashtags for context
        for other_tag in other_hashtags:
            other_clean = other_tag.replace("#", "").lower()
            for cluster_name, cluster_info in semantic_clusters.items():
                for keyword in cluster_info["keywords"]:
                    if keyword in other_clean:
                        return cluster_name

        return "general"  # Default cluster

    # 3. Determine semantic cluster for this hashtag
    cluster = get_semantic_cluster(hashtag)

    # 4. Assign semantic clusters to partition ranges
    if cluster in semantic_clusters:
        start_partition, end_partition = semantic_clusters[cluster]["partition_range"]
        partition_range = max(1, end_partition - start_partition)

        # For high-engagement content, add some distribution within cluster
        if engagement_value >= 100:
            # High engagement: use full cluster range with hash distribution
            return start_partition + (hash(hashtag) % partition_range)
        else:
            # Normal engagement: more concentrated within cluster
            concentrated_range = max(1, partition_range // 2)
            return start_partition + (hash(hashtag) % concentrated_range)

    else:
        # General/unclassified hashtags: use remaining partitions
        remaining_partitions = max(1, num_partitions // 4)  # Reserve 25% for general
        general_start = num_partitions - remaining_partitions
        return general_start + (hash(hashtag) % remaining_partitions)


def measure_hashtag_balance(partition_results: Dict[int, Dict[str, Dict[str, any]]]) -> Dict[str, float]:
    """
    Calculate metrics to measure how well-balanced the partitions are for hashtag analysis.
    """
    import math

    # 1. Calculate load per partition (number of hashtags)
    per_partition_hashtag_count = {}
    per_partition_trending_score = {}
    per_partition_frequency_sum = {}
    all_co_occurrences = {}

    for partition_id, hashtag_metrics in partition_results.items():
        hashtag_count = len(hashtag_metrics)
        trending_sum = sum(metrics.get('trending_score', 0) for metrics in hashtag_metrics.values())
        frequency_sum = sum(metrics.get('frequency', 0) for metrics in hashtag_metrics.values())

        per_partition_hashtag_count[partition_id] = hashtag_count
        per_partition_trending_score[partition_id] = trending_sum
        per_partition_frequency_sum[partition_id] = frequency_sum

        # Collect all co-occurrence data for locality analysis
        for hashtag, metrics in hashtag_metrics.items():
            co_occurrences = metrics.get('co_occurrences', {})
            all_co_occurrences[hashtag] = {
                'partition': partition_id,
                'co_occurrences': co_occurrences
            }

    # Handle empty partitions
    if not per_partition_hashtag_count:
        return {
            "hashtag_balance_ratio": 0.0,
            "trending_balance_ratio": 0.0,
            "frequency_balance_ratio": 0.0,
            "std_dev": 0.0,
            "gini_coefficient": 0.0,
            "co_occurrence_locality": 0.0
        }

    # 2. Calculate balance ratios
    hashtag_counts = list(per_partition_hashtag_count.values())
    trending_scores = list(per_partition_trending_score.values())
    frequency_sums = list(per_partition_frequency_sum.values())

    # Hashtag count balance (max_load / avg_load)
    avg_hashtag_count = sum(hashtag_counts) / len(hashtag_counts) if hashtag_counts else 0
    hashtag_balance_ratio = max(hashtag_counts) / avg_hashtag_count if avg_hashtag_count > 0 else 0

    # Trending score balance
    avg_trending_score = sum(trending_scores) / len(trending_scores) if trending_scores else 0
    trending_balance_ratio = max(trending_scores) / avg_trending_score if avg_trending_score > 0 else 0

    # Frequency balance
    avg_frequency = sum(frequency_sums) / len(frequency_sums) if frequency_sums else 0
    frequency_balance_ratio = max(frequency_sums) / avg_frequency if avg_frequency > 0 else 0

    # 3. Standard deviation of hashtag loads
    variance = sum((count - avg_hashtag_count) ** 2 for count in hashtag_counts) / len(hashtag_counts) if hashtag_counts else 0
    std_dev = math.sqrt(variance)

    # 4. Gini coefficient for inequality measurement
    sorted_counts = sorted(hashtag_counts)
    n = len(sorted_counts)
    if n > 0 and sum(sorted_counts) > 0:
        weighted_sum = sum((i + 1) * count for i, count in enumerate(sorted_counts))
        gini_coefficient = (2 * weighted_sum) / (n * sum(sorted_counts)) - (n + 1) / n
    else:
        gini_coefficient = 0.0

    # 5. Co-occurrence locality analysis
    total_co_occurrence_pairs = 0
    local_co_occurrence_pairs = 0

    for hashtag, data in all_co_occurrences.items():
        hashtag_partition = data['partition']
        co_occurrences = data['co_occurrences']

        for co_hashtag, count in co_occurrences.items():
            if co_hashtag in all_co_occurrences:
                total_co_occurrence_pairs += count
                co_hashtag_partition = all_co_occurrences[co_hashtag]['partition']

                # Check if co-occurring hashtags are in the same partition
                if hashtag_partition == co_hashtag_partition:
                    local_co_occurrence_pairs += count

    co_occurrence_locality = local_co_occurrence_pairs / total_co_occurrence_pairs if total_co_occurrence_pairs > 0 else 1.0

    return {
        "hashtag_balance_ratio": hashtag_balance_ratio,
        "trending_balance_ratio": trending_balance_ratio,
        "frequency_balance_ratio": frequency_balance_ratio,
        "std_dev": std_dev,
        "gini_coefficient": gini_coefficient,
        "co_occurrence_locality": co_occurrence_locality,
        "total_hashtags": sum(hashtag_counts),
        "num_partitions": len(per_partition_hashtag_count)
    }


def analyze_hashtag_co_occurrence(partition_results: Dict[int, Dict[str, Dict[str, any]]]) -> Dict[str, any]:
    """
    Analyze co-occurrence patterns and measure partitioning effectiveness for co-occurrence queries.
    """
    # 1. Extract co-occurrence matrices from all partitions
    global_co_occurrence_matrix = defaultdict(lambda: defaultdict(int))
    hashtag_partition_map = {}
    partition_co_occurrence_counts = defaultdict(int)

    for partition_id, hashtag_metrics in partition_results.items():
        for hashtag, metrics in hashtag_metrics.items():
            hashtag_partition_map[hashtag] = partition_id
            co_occurrences = metrics.get('co_occurrences', {})

            # Build global co-occurrence matrix
            for co_hashtag, count in co_occurrences.items():
                global_co_occurrence_matrix[hashtag][co_hashtag] += count
                partition_co_occurrence_counts[partition_id] += count

    # 2. Measure cross-partition co-occurrence queries
    total_co_occurrence_relationships = 0
    local_co_occurrence_relationships = 0
    cross_partition_relationships = 0

    most_connected_hashtags = []
    cross_partition_queries = []

    for hashtag, co_occurrences in global_co_occurrence_matrix.items():
        if hashtag not in hashtag_partition_map:
            continue

        hashtag_partition = hashtag_partition_map[hashtag]
        total_connections = sum(co_occurrences.values())
        local_connections = 0
        cross_connections = 0

        for co_hashtag, count in co_occurrences.items():
            if co_hashtag in hashtag_partition_map:
                total_co_occurrence_relationships += count
                co_hashtag_partition = hashtag_partition_map[co_hashtag]

                if hashtag_partition == co_hashtag_partition:
                    local_co_occurrence_relationships += count
                    local_connections += count
                else:
                    cross_partition_relationships += count
                    cross_connections += count
                    cross_partition_queries.append({
                        'hashtag1': hashtag,
                        'hashtag2': co_hashtag,
                        'partition1': hashtag_partition,
                        'partition2': co_hashtag_partition,
                        'count': count
                    })

        # Track most connected hashtags
        if total_connections > 0:
            most_connected_hashtags.append({
                'hashtag': hashtag,
                'partition': hashtag_partition,
                'total_connections': total_connections,
                'local_connections': local_connections,
                'cross_connections': cross_connections,
                'locality_ratio': local_connections / total_connections
            })

    # 3. Calculate co-occurrence locality ratio
    locality_ratio = local_co_occurrence_relationships / total_co_occurrence_relationships if total_co_occurrence_relationships > 0 else 1.0

    # 4. Identify most connected hashtag clusters
    # Sort by total connections and locality
    most_connected_hashtags.sort(key=lambda x: x['total_connections'], reverse=True)
    top_connected = most_connected_hashtags[:10]  # Top 10 most connected

    # Find hashtags with poor locality (high cross-partition connections)
    poor_locality_hashtags = [h for h in most_connected_hashtags
                             if h['total_connections'] >= 5 and h['locality_ratio'] < 0.5]
    poor_locality_hashtags.sort(key=lambda x: x['locality_ratio'])

    # Analyze partition connectivity
    partition_cross_connections = defaultdict(int)
    for query in cross_partition_queries:
        partition_cross_connections[query['partition1']] += query['count']
        partition_cross_connections[query['partition2']] += query['count']

    # 5. Return co-occurrence analysis metrics
    return {
        'locality_ratio': locality_ratio,
        'total_co_occurrence_relationships': total_co_occurrence_relationships,
        'local_relationships': local_co_occurrence_relationships,
        'cross_partition_relationships': cross_partition_relationships,
        'cross_partition_query_ratio': cross_partition_relationships / total_co_occurrence_relationships if total_co_occurrence_relationships > 0 else 0.0,
        'most_connected_hashtags': top_connected,
        'poor_locality_hashtags': poor_locality_hashtags[:5],  # Top 5 problematic
        'partition_cross_connections': dict(partition_cross_connections),
        'total_unique_hashtags': len(hashtag_partition_map),
        'average_connections_per_hashtag': sum(len(co_occ) for co_occ in global_co_occurrence_matrix.values()) / len(global_co_occurrence_matrix) if global_co_occurrence_matrix else 0
    }


def run_hashtag_experiment():
    """
    Run experiments comparing different partitioning strategies for hashtag trending analysis.
    """
    print("🧪 PARTITIONING EXPERIMENT: Hashtag Trending Analysis")
    print("="*60)

    # Define partitioning strategies to test
    strategies = [
        PartitioningStrategy("hash", hash_partition, 8),
        PartitioningStrategy("popularity_tier", popularity_tier_partition, 8),
        PartitioningStrategy("co_occurrence", co_occurrence_partition, 8),
        PartitioningStrategy("semantic_cluster", semantic_cluster_partition, 8),
    ]

    # Look for hashtag data files (could be in content engagement or separate hashtag files)
    data_files = list(Path("social_media_data").glob("*hashtag*.jsonl"))
    if not data_files:
        # Fallback to content engagement files which contain hashtags
        data_files = list(Path("social_media_data").glob("content_engagement_*.jsonl"))

    if not data_files:
        print("❌ No hashtag data files found. Run data_generator.py first.")
        return

    results = {}

    for strategy in strategies:
        print(f"\n🔄 Testing {strategy.name} partitioning...")

        start_time = time.time()

        # 1. Process files with the partitioning strategy
        mapreduce_class = HashtagTrendingMapReduce
        per_partition_data = defaultdict(list)

        for file_path in data_files:
            with open(file_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    per_line_data = mapreduce_class.partition_aware_map(line, strategy)
                    for partition_id, hashtag_data in per_line_data:
                        per_partition_data[partition_id].append(hashtag_data)

        reduced_data = mapreduce_class.partition_aware_reduce(per_partition_data)

        # 2. Measure load balance and trending distribution
        hashtag_balance = measure_hashtag_balance(reduced_data)

        # 3. Analyze co-occurrence locality
        co_occurrence_analysis = analyze_hashtag_co_occurrence(reduced_data)

        # 4. Calculate performance metrics
        processing_time = time.time() - start_time

        # Store results for comparison
        results[strategy.name] = {
            'processing_time': processing_time,
            **hashtag_balance,
            'co_occurrence_locality_detailed': co_occurrence_analysis['locality_ratio'],
            'cross_partition_query_ratio': co_occurrence_analysis['cross_partition_query_ratio'],
            'most_connected_count': len(co_occurrence_analysis['most_connected_hashtags']),
            'poor_locality_count': len(co_occurrence_analysis['poor_locality_hashtags'])
        }

    # Print comparison results
    print("\n📊 EXPERIMENT RESULTS")
    print("-" * 40)
    print("Strategy | Processing Time | Hashtag Balance | Trending Balance | Co-occurrence Locality | Std Dev | Gini Score")
    print("-"*120)
    for strategy, metrics in results.items():
        print(f"{strategy:<15} | {metrics['processing_time']:.4f} | {metrics['hashtag_balance_ratio']:.4f} | {metrics['trending_balance_ratio']:.4f} | {metrics['co_occurrence_locality']:.4f} | {metrics['std_dev']:.4f} | {metrics['gini_coefficient']:.4f}")

    # Detailed co-occurrence analysis
    print("\n🔗 CO-OCCURRENCE ANALYSIS")
    print("-" * 40)
    for strategy, metrics in results.items():
        print(f"\n{strategy} Strategy:")
        print(f"  Cross-partition queries: {metrics['cross_partition_query_ratio']:.2%}")
        print(f"  Most connected hashtags: {metrics['most_connected_count']}")
        print(f"  Poor locality hashtags: {metrics['poor_locality_count']}")

    # Find best strategy
    best_strategy = min(results.keys(),
                       key=lambda x: results[x]['cross_partition_query_ratio'] +
                                   (1 - results[x]['co_occurrence_locality']) +
                                   results[x]['gini_coefficient'])

    print(f"\n🏆 BEST STRATEGY: {best_strategy}")
    print(f"Optimizes for: Low cross-partition queries + High co-occurrence locality + Low inequality")

    return results


def visualize_hashtag_trends():
    """
    Create visualizations showing hashtag trending patterns and co-occurrence networks.

    Implement hashtag trend visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 1. Read hashtag data
    data_files = list(Path("social_media_data").glob("*hashtag*.jsonl"))
    if not data_files:
        data_files = list(Path("social_media_data").glob("content_engagement_*.jsonl"))

    if not data_files:
        print("❌ No hashtag data files found.")
        return

    mapreduce_class = HashtagTrendingMapReduce
    per_file_data = []

    for file in data_files:
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                per_line_data = mapreduce_class.map(line)
                for hashtag_data in per_line_data:
                    per_file_data.append(hashtag_data)

    # 2. Calculate trending scores
    per_hashtag_metrics = mapreduce_class.reduce([iter([data]) for data in per_file_data])

    if not per_hashtag_metrics:
        print("❌ No hashtag data processed.")
        return

    # Extract metrics for visualization
    hashtags = list(per_hashtag_metrics.keys())
    trending_scores = [metrics['trending_score'] for metrics in per_hashtag_metrics.values()]
    frequencies = [metrics['frequency'] for metrics in per_hashtag_metrics.values()]

    # 3. Create plots showing:

    # Hashtag frequency distribution (power law)
    plt.figure(figsize=(15, 12))

    plt.subplot(2, 3, 1)
    plt.hist(frequencies, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title("Hashtag Frequency Distribution", fontsize=14, fontweight='bold')
    plt.xlabel("Frequency Count", fontsize=12)
    plt.ylabel("Number of Hashtags", fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    # Top trending hashtags (bar chart)
    plt.subplot(2, 3, 2)
    top_hashtags = sorted(per_hashtag_metrics.items(), key=lambda x: x[1]['trending_score'], reverse=True)[:15]
    top_names = [item[0][:10] for item in top_hashtags]  # Truncate long hashtags
    top_scores = [item[1]['trending_score'] for item in top_hashtags]

    plt.barh(range(len(top_names)), top_scores, color='orange', alpha=0.8)
    plt.yticks(range(len(top_names)), top_names, fontsize=10)
    plt.title("Top 15 Trending Hashtags", fontsize=14, fontweight='bold')
    plt.xlabel("Trending Score", fontsize=12)
    plt.grid(True, alpha=0.3)

    # Trending vs Frequency scatter
    plt.subplot(2, 3, 3)
    plt.scatter(frequencies, trending_scores, alpha=0.6, s=30, color='purple')
    plt.title("Trending Score vs Frequency", fontsize=14, fontweight='bold')
    plt.xlabel("Frequency Count", fontsize=12)
    plt.ylabel("Trending Score", fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    # Temporal trending patterns by engagement type
    plt.subplot(2, 3, 4)
    engagement_trending = {"share": [], "comment": [], "like": [], "save": [], "view": []}

    for file in data_files:
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                record = json.loads(line.strip())
                hashtags = record.get("hashtags", [])
                engagement_type = record["engagement_type"]

                for hashtag in hashtags:
                    if hashtag in per_hashtag_metrics:
                        trending_score = per_hashtag_metrics[hashtag]['trending_score']
                        engagement_trending[engagement_type].append(trending_score)

    plt.boxplot([engagement_trending["share"], engagement_trending["comment"], engagement_trending["like"], engagement_trending["view"], engagement_trending["save"]],
                labels=["Share", "Comment", "Like", "View", "Save"])
    plt.title("Trending Patterns by Engagement Type", fontsize=14, fontweight='bold')
    plt.xlabel("Engagement Type", fontsize=12)
    plt.ylabel("Trending Score", fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    # Co-occurrence network visualization (simplified)
    plt.subplot(2, 3, 5)
    # Get top connected hashtags for network visualization
    co_occurrence_counts = []
    hashtag_connections = {}

    for hashtag, metrics in per_hashtag_metrics.items():
        co_occurrences = metrics.get('co_occurrences', {})
        total_connections = sum(co_occurrences.values())
        co_occurrence_counts.append(total_connections)
        hashtag_connections[hashtag] = total_connections

    plt.hist(co_occurrence_counts, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.title("Hashtag Co-occurrence Distribution", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Co-occurrences", fontsize=12)
    plt.ylabel("Number of Hashtags", fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    # Partition load distribution comparison
    plt.subplot(2, 3, 6)
    strategies = [
        PartitioningStrategy("hash", hash_partition, 8),
        PartitioningStrategy("co_occurrence", co_occurrence_partition, 8),
        PartitioningStrategy("semantic_cluster", semantic_cluster_partition, 8),
    ]

    strategy_loads = {}
    for strategy in strategies:
        per_partition_data = defaultdict(list)
        for file_path in data_files:
            with open(file_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    per_line_data = mapreduce_class.partition_aware_map(line, strategy)
                    for partition_id, hashtag_data in per_line_data:
                        per_partition_data[partition_id].append(hashtag_data)

        partition_loads = [len(per_partition_data[pid]) for pid in range(8)]
        strategy_loads[strategy.name] = partition_loads

    x = np.arange(8)
    width = 0.25
    for i, (strategy_name, loads) in enumerate(strategy_loads.items()):
        plt.bar(x + i * width, loads, width, label=strategy_name, alpha=0.8)

    plt.title("Partition Load Distribution Comparison", fontsize=14, fontweight='bold')
    plt.xlabel("Partition ID", fontsize=12)
    plt.ylabel("Number of Hashtag Instances", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.suptitle("Hashtag Trending Analysis Visualizations", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Summary statistics
    print(f"\n📊 Hashtag Analysis Summary:")
    print(f"Total unique hashtags: {len(hashtags)}")
    print(f"Average trending score: {np.mean(trending_scores):.2f}")
    print(f"Max trending score: {max(trending_scores):.2f}")
    print(f"Average frequency: {np.mean(frequencies):.2f}")
    print(f"Average co-occurrences per hashtag: {np.mean(co_occurrence_counts):.2f}")
    print(f"Most connected hashtag: {max(hashtag_connections.items(), key=lambda x: x[1])}")

    return per_hashtag_metrics


def analyze_trending_hotspots():
    """
    Identify and analyze trending hashtag hotspots that cause partitioning issues.

    Implement trending hotspot analysis
    """
    print("🔥 TRENDING HASHTAG HOTSPOT ANALYSIS")
    print("="*50)

    # Read hashtag data
    data_files = list(Path("social_media_data").glob("*hashtag*.jsonl"))
    if not data_files:
        data_files = list(Path("social_media_data").glob("content_engagement_*.jsonl"))

    if not data_files:
        print("❌ No hashtag data files found.")
        return

    mapreduce_class = HashtagTrendingMapReduce
    per_file_data = []

    for file in data_files:
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                per_line_data = mapreduce_class.map(line)
                for hashtag_data in per_file_data:
                    per_file_data.append(hashtag_data)

    # Calculate trending scores
    per_hashtag_metrics = mapreduce_class.reduce([iter([data]) for data in per_file_data])

    if not per_hashtag_metrics:
        print("❌ No hashtag data processed.")
        return

    # 1. Identify hashtags that exceed trending thresholds
    trending_scores = [metrics['trending_score'] for metrics in per_hashtag_metrics.values()]
    frequencies = [metrics['frequency'] for metrics in per_hashtag_metrics.values()]

    avg_trending = sum(trending_scores) / len(trending_scores)
    avg_frequency = sum(frequencies) / len(frequencies)

    # Define hotspot thresholds
    high_threshold = avg_trending * 3     # 3x average = hot
    extreme_threshold = avg_trending * 5  # 5x average = extreme hotspot

    hot_hashtags = {tag: metrics for tag, metrics in per_hashtag_metrics.items()
                   if metrics['trending_score'] >= high_threshold}
    extreme_hashtags = {tag: metrics for tag, metrics in per_hashtag_metrics.items()
                       if metrics['trending_score'] >= extreme_threshold}

    print(f"\n📊 Hotspot Identification:")
    print(f"Average trending score: {avg_trending:.2f}")
    print(f"Hot hashtag threshold (3x avg): {high_threshold:.2f}")
    print(f"Extreme threshold (5x avg): {extreme_threshold:.2f}")
    print(f"Hot hashtags: {len(hot_hashtags)} ({len(hot_hashtags)/len(per_hashtag_metrics)*100:.1f}%)")
    print(f"Extreme hashtags: {len(extreme_hashtags)} ({len(extreme_hashtags)/len(per_hashtag_metrics)*100:.1f}%)")

    # 2. Analyze temporal clustering of trending events
    print(f"\n⏰ Temporal Clustering Analysis:")

    trending_by_type = {"share": [], "comment": [], "like": []}
    hot_by_type = {"share": 0, "comment": 0, "like": 0}

    for file in data_files:
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                record = json.loads(line.strip())
                hashtags = record.get("hashtags", [])
                engagement_type = record["engagement_type"]

                for hashtag in hashtags:
                    if hashtag in per_hashtag_metrics:
                        trending_score = per_hashtag_metrics[hashtag]['trending_score']
                        trending_by_type[engagement_type].append(trending_score)

                        if trending_score >= high_threshold:
                            hot_by_type[engagement_type] += 1

    for eng_type in trending_by_type:
        if trending_by_type[eng_type]:
            avg_by_type = sum(trending_by_type[eng_type]) / len(trending_by_type[eng_type])
            print(f"{eng_type.capitalize()} events - Avg trending: {avg_by_type:.2f}, Hot hashtags: {hot_by_type[eng_type]}")

    # 3. Measure impact on partition balance
    print(f"\n⚖️ Partition Balance Impact Analysis:")

    strategies = [
        PartitioningStrategy("hash", hash_partition, 8),
        PartitioningStrategy("popularity_tier", popularity_tier_partition, 8),
        PartitioningStrategy("co_occurrence", co_occurrence_partition, 8),
        PartitioningStrategy("semantic_cluster", semantic_cluster_partition, 8),
    ]

    balance_impact = {}

    for strategy in strategies:
        per_partition_data = defaultdict(list)
        for file_path in data_files:
            with open(file_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    per_line_data = mapreduce_class.partition_aware_map(line, strategy)
                    for partition_id, hashtag_data in per_line_data:
                        per_partition_data[partition_id].append(hashtag_data)

        reduced_data = mapreduce_class.partition_aware_reduce(per_partition_data)

        # Find which partitions contain hot hashtags
        hot_partitions = set()
        hotspot_concentration = {}

        for pid, hashtag_dict in reduced_data.items():
            partition_hot_count = 0
            partition_hot_trending = 0

            for hashtag, metrics in hashtag_dict.items():
                trending_score = metrics.get('trending_score', 0)
                if trending_score >= high_threshold:
                    hot_partitions.add(pid)
                    partition_hot_count += 1
                    partition_hot_trending += trending_score

            if partition_hot_count > 0:
                hotspot_concentration[pid] = {
                    'hot_count': partition_hot_count,
                    'hot_trending': partition_hot_trending,
                    'total_hashtags': len(hashtag_dict)
                }

        # Calculate balance metrics
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

    # 4. Analyze co-occurrence locality for trending hashtags
    print(f"\n🔗 Co-occurrence Locality for Trending Hashtags:")

    trending_locality_issues = []
    for hashtag, metrics in hot_hashtags.items():
        co_occurrences = metrics.get('co_occurrences', {})
        if co_occurrences:
            # Check how many co-occurring hashtags are also trending
            trending_co_occurrences = sum(1 for co_tag in co_occurrences.keys()
                                        if co_tag in hot_hashtags)
            total_co_occurrences = len(co_occurrences)

            if trending_co_occurrences > 2:  # Multiple trending hashtags co-occur
                trending_locality_issues.append({
                    'hashtag': hashtag,
                    'trending_score': metrics['trending_score'],
                    'trending_co_occurrences': trending_co_occurrences,
                    'total_co_occurrences': total_co_occurrences
                })

    trending_locality_issues.sort(key=lambda x: x['trending_co_occurrences'], reverse=True)

    print(f"Found {len(trending_locality_issues)} hashtags with multiple trending co-occurrences")
    for issue in trending_locality_issues[:5]:
        print(f"  {issue['hashtag']}: {issue['trending_co_occurrences']}/{issue['total_co_occurrences']} "
              f"trending co-occurrences (score: {issue['trending_score']:.2f})")

    # 5. Suggest mitigation strategies for hashtag skew
    print(f"\n💡 Mitigation Strategy Recommendations:")

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

    if len(extreme_hashtags) > 0:
        print(f"1. CRITICAL: {len(extreme_hashtags)} extreme trending hashtags detected!")
        print(f"   - Implement dedicated processing pools for hashtags >5x average trending")
        print(f"   - Consider real-time trending detection and load redistribution")

    if balance_impact[worst_strategy]['hot_imbalance'] > 3.0:
        print(f"2. HIGH: {worst_strategy} partitioning shows severe imbalance (>3x)")
        print(f"   - Avoid {worst_strategy} for trending hashtag workloads")
        print(f"   - Implement dynamic load balancing for trending events")

    if len(trending_locality_issues) > 5:
        print(f"3. LOCALITY: {len(trending_locality_issues)} hashtags have trending co-occurrence clusters")
        print(f"   - Use co-occurrence aware partitioning for trending hashtag groups")
        print(f"   - Implement trending cluster detection")

    if hot_by_type['share'] > hot_by_type['like'] * 2:
        print(f"4. TEMPORAL: Share events show higher trending concentration")
        print(f"   - Implement recency-based load balancing")
        print(f"   - Use temporal partitioning for recent trending events")

    print(f"5. GENERAL: Use {best_strategy} partitioning for better trending hashtag distribution")
    print(f"6. MONITORING: Set up alerts for hashtags exceeding {high_threshold:.1f} trending score")

    return {
        'hot_hashtags': len(hot_hashtags),
        'extreme_hashtags': len(extreme_hashtags),
        'best_strategy': best_strategy,
        'worst_strategy': worst_strategy,
        'balance_impact': balance_impact,
        'trending_locality_issues': len(trending_locality_issues)
    }


def build_hashtag_network():
    """
    Build and analyze the hashtag co-occurrence network for partitioning insights.

    Analyzes hashtag co-occurrence patterns to identify clusters and recommend
    network topology-based partitioning strategies.
    """
    print("🕸️ HASHTAG NETWORK ANALYSIS")
    print("="*50)

    # Get data files
    data_files = list(Path("social_media_data").glob("*hashtag*.jsonl"))
    if not data_files:
        data_files = list(Path("social_media_data").glob("content_engagement_*.jsonl"))


    mapreduce_class = HashtagTrendingMapReduce()

    # 1. Build hashtag co-occurrence graph from all partitioning strategies
    print("🔗 Building Co-occurrence Network...")

    # Initialize network data structures
    co_occurrence_graph = defaultdict(lambda: defaultdict(int))
    hashtag_frequencies = defaultdict(int)

    strategies = [
        PartitioningStrategy("hash", hash_partition, 8),
        PartitioningStrategy("popularity_tier", popularity_tier_partition, 8),
        PartitioningStrategy("co_occurrence", co_occurrence_partition, 8),
        PartitioningStrategy("semantic_cluster", semantic_cluster_partition, 8),
    ]

    # Use co-occurrence strategy as our primary network source
    strategy = strategies[2]  # co_occurrence strategy

    per_partition_data = defaultdict(list)
    for file_path in data_files:
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                per_line_data = mapreduce_class.partition_aware_map(line, strategy)
                for partition_id, hashtag_data in per_line_data:
                    per_partition_data[partition_id].append(hashtag_data)

    reduced_data = mapreduce_class.partition_aware_reduce(per_partition_data)

    # Build network from reduced data
    for partition_id, hashtag_dict in reduced_data.items():
        for hashtag, metrics in hashtag_dict.items():
            hashtag_frequencies[hashtag] = metrics.get('frequency', 0)
            co_occurrences = metrics.get('co_occurrences', {})

            for co_hashtag, co_count in co_occurrences.items():
                # Build undirected graph
                co_occurrence_graph[hashtag][co_hashtag] += co_count
                co_occurrence_graph[co_hashtag][hashtag] += co_count

    print(f"Network built: {len(co_occurrence_graph)} nodes, {sum(len(edges) for edges in co_occurrence_graph.values())//2} edges")

    # 2. Identify highly connected hashtag clusters
    print(f"\n🎯 Identifying Hashtag Clusters...")

    # Calculate node degree and connectivity metrics
    node_degrees = {}
    for hashtag, connections in co_occurrence_graph.items():
        degree = len(connections)
        total_weight = sum(connections.values())
        node_degrees[hashtag] = {
            'degree': degree,
            'weighted_degree': total_weight,
            'avg_edge_weight': total_weight / degree if degree > 0 else 0
        }

    # Find high-degree nodes (hubs)
    if node_degrees:
        avg_degree = sum(metrics['degree'] for metrics in node_degrees.values()) / len(node_degrees)
        high_degree_threshold = avg_degree * 2

        hub_hashtags = {hashtag: metrics for hashtag, metrics in node_degrees.items()
                       if metrics['degree'] >= high_degree_threshold}

        print(f"Average node degree: {avg_degree:.2f}")
        print(f"Hub threshold (2x avg): {high_degree_threshold:.2f}")
        print(f"Identified {len(hub_hashtags)} hub hashtags:")

        # Sort hubs by degree
        sorted_hubs = sorted(hub_hashtags.items(), key=lambda x: x[1]['degree'], reverse=True)
        for hashtag, metrics in sorted_hubs[:10]:  # Top 10 hubs
            print(f"  {hashtag}: {metrics['degree']} connections, "
                  f"weight: {metrics['weighted_degree']}, "
                  f"avg: {metrics['avg_edge_weight']:.2f}")

    # 3. Analyze community structure using simple clustering
    print(f"\n🏘️ Community Structure Analysis...")

    # Simple clustering: group hashtags by shared high-weight connections
    clusters = []
    visited = set()

    def find_cluster(start_hashtag, min_edge_weight=2):
        cluster = {start_hashtag}
        queue = [start_hashtag]

        while queue:
            current = queue.pop(0)
            if current in co_occurrence_graph:
                for neighbor, weight in co_occurrence_graph[current].items():
                    if neighbor not in visited and weight >= min_edge_weight:
                        visited.add(neighbor)
                        cluster.add(neighbor)
                        queue.append(neighbor)

        return cluster

    # Find clusters starting from high-degree nodes
    for hashtag in sorted(node_degrees.keys(), key=lambda x: node_degrees[x]['degree'], reverse=True):
        if hashtag not in visited:
            visited.add(hashtag)
            cluster = find_cluster(hashtag)
            if len(cluster) > 1:  # Only keep non-trivial clusters
                clusters.append(cluster)

    # Sort clusters by size
    clusters.sort(key=len, reverse=True)

    print(f"Found {len(clusters)} communities:")
    for i, cluster in enumerate(clusters[:5]):  # Top 5 communities
        print(f"  Community {i+1}: {len(cluster)} hashtags")
        cluster_list = list(cluster)
        if len(cluster_list) <= 5:
            print(f"    Members: {', '.join(cluster_list)}")
        else:
            print(f"    Sample: {', '.join(cluster_list[:5])}...")

    # 4. Recommend partitioning based on network topology
    print(f"\n📊 Network-Based Partitioning Recommendations...")

    # Measure cross-partition connectivity for each strategy
    cross_partition_analysis = {}

    for strategy in strategies:
        per_partition_data = defaultdict(list)

        # Re-partition data with this strategy
        for file_path in data_files:
            with open(file_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    per_line_data = mapreduce_class.partition_aware_map(line, strategy)
                    for partition_id, hashtag_data in per_line_data:
                        per_partition_data[partition_id].append(hashtag_data)

        # Calculate cross-partition edges
        hashtag_to_partition = {}
        for partition_id, hashtag_list in per_partition_data.items():
            for hashtag_data in hashtag_list:
                hashtag = hashtag_data['hashtag']
                hashtag_to_partition[hashtag] = partition_id

        intra_partition_edges = 0
        cross_partition_edges = 0
        total_edge_weight = 0
        cross_edge_weight = 0

        for hashtag, connections in co_occurrence_graph.items():
            if hashtag in hashtag_to_partition:
                hashtag_partition = hashtag_to_partition[hashtag]

                for connected_hashtag, weight in connections.items():
                    if connected_hashtag in hashtag_to_partition:
                        connected_partition = hashtag_to_partition[connected_hashtag]
                        total_edge_weight += weight

                        if hashtag_partition == connected_partition:
                            intra_partition_edges += 1
                        else:
                            cross_partition_edges += 1
                            cross_edge_weight += weight

        total_edges = intra_partition_edges + cross_partition_edges
        cross_partition_ratio = cross_partition_edges / total_edges if total_edges > 0 else 0
        cross_weight_ratio = cross_edge_weight / total_edge_weight if total_edge_weight > 0 else 0

        cross_partition_analysis[strategy.name] = {
            'cross_partition_ratio': cross_partition_ratio,
            'cross_weight_ratio': cross_weight_ratio,
            'total_edges': total_edges,
            'cross_edges': cross_partition_edges
        }

        print(f"{strategy.name}:")
        print(f"  Cross-partition edges: {cross_partition_edges}/{total_edges} ({cross_partition_ratio*100:.1f}%)")
        print(f"  Cross-partition weight: {cross_weight_ratio*100:.1f}%")

    # 5. Measure cluster distribution across partitions
    print(f"\n🔍 Cluster Distribution Analysis...")

    best_strategy_name = min(cross_partition_analysis.keys(),
                           key=lambda x: cross_partition_analysis[x]['cross_partition_ratio'])
    worst_strategy_name = max(cross_partition_analysis.keys(),
                            key=lambda x: cross_partition_analysis[x]['cross_partition_ratio'])

    print(f"✅ Best locality: {best_strategy_name}")
    print(f"   - Lowest cross-partition ratio: {cross_partition_analysis[best_strategy_name]['cross_partition_ratio']*100:.1f}%")

    print(f"❌ Worst locality: {worst_strategy_name}")
    print(f"   - Highest cross-partition ratio: {cross_partition_analysis[worst_strategy_name]['cross_partition_ratio']*100:.1f}%")

    # Network-based recommendations
    print(f"\n🎯 Network Topology Recommendations:")

    if len(hub_hashtags) > 0:
        print(f"1. HUB MANAGEMENT: {len(hub_hashtags)} hub hashtags detected")
        print(f"   - Replicate hub hashtags across multiple partitions")
        print(f"   - Use dedicated processing for high-degree nodes")

    if len(clusters) > 8:  # More clusters than partitions
        print(f"2. CLUSTER ALIGNMENT: {len(clusters)} communities found (>8 partitions)")
        print(f"   - Use community-aware partitioning")
        print(f"   - Consider hierarchical partitioning")

    avg_cross_ratio = sum(data['cross_partition_ratio'] for data in cross_partition_analysis.values()) / len(cross_partition_analysis)
    if avg_cross_ratio > 0.3:  # High cross-partition connectivity
        print(f"3. LOCALITY OPTIMIZATION: High cross-partition connectivity ({avg_cross_ratio*100:.1f}%)")
        print(f"   - Implement edge-cut minimization")
        print(f"   - Use graph partitioning algorithms (e.g., METIS)")

    if cross_partition_analysis[best_strategy_name]['cross_partition_ratio'] < 0.2:
        print(f"4. OPTIMAL: {best_strategy_name} shows excellent locality (<20% cross-partition)")
        print(f"   - Recommended for hashtag co-occurrence workloads")

    print(f"5. MONITORING: Track network metrics for evolving hashtag relationships")
    print(f"6. CACHING: Cache frequently accessed cross-partition hashtag pairs")

    return {
        'total_nodes': len(co_occurrence_graph),
        'total_edges': sum(len(edges) for edges in co_occurrence_graph.values()) // 2,
        'hub_hashtags': len(hub_hashtags) if 'hub_hashtags' in locals() else 0,
        'communities': len(clusters),
        'best_locality_strategy': best_strategy_name,
        'worst_locality_strategy': worst_strategy_name,
        'cross_partition_analysis': cross_partition_analysis
    }


if __name__ == "__main__":
    print("🎯 CHALLENGE 3: Hashtag Trending Analysis with Partitioning")
    print("="*60)
    print()


    run_hashtag_experiment()
    visualize_hashtag_trends()
    analyze_trending_hotspots()
    #build_hashtag_network()
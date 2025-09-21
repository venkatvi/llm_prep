"""
Challenge 4: Regional Activity Patterns with Geographic Partitioning

This challenge focuses on analyzing user behavior patterns by geographic region,
handling geographic data skew and optimizing data locality for cross-regional
influence analysis.

Problem: Analyze user activity patterns across different geographic regions,
but handle the fact that some regions have much higher activity than others,
and cross-regional analysis requires efficient data shuffling.

Challenges:
- Geographic data skew (some regions much more active)
- Cross-regional influence analysis requires data shuffling
- Time zone considerations create temporal skew
- Need data locality optimization for regional analysis
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
import datetime


@dataclass
class PartitioningStrategy:
    """Configuration for a partitioning strategy."""
    name: str
    partition_func: callable
    num_partitions: int


class RegionalActivityMapReduce:
    """
    MapReduce implementation for analyzing regional activity patterns
    with different geographic partitioning strategies.
    """

    @staticmethod
    def map(line: str) -> Generator[Tuple[str, Dict], None, None]:
        """
        Map phase: Extract regional activity data from user records.

        Parse JSON line and extract region, user_id, activity metrics
        """
        record = json.loads(line.strip())

        # Extract geographic and temporal information
        region = record.get("region", "unknown")
        user_id = str(record.get("user_id", ""))
        timestamp = record.get("timestamp", 0)
        activity_type = record.get("activity_type", "unknown")
        engagement_score = record.get("engagement_score", 0)

        # Calculate hour of day for temporal analysis
        dt = datetime.datetime.fromtimestamp(timestamp)
        hour_of_day = dt.hour
        day_of_week = dt.weekday()  # 0=Monday, 6=Sunday

        # Create composite key for regional analysis
        regional_key = f"{region}"

        # Emit regional activity data
        activity_data = {
            'user_id': user_id,
            'region': region,
            'timestamp': timestamp,
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
            'activity_type': activity_type,
            'engagement_score': engagement_score,
            'activity_count': 1
        }

        yield (regional_key, activity_data)

    @staticmethod
    def partition_aware_map(line: str, strategy: PartitioningStrategy) -> List[Tuple[int, Dict]]:
        """
        Partition-aware map that assigns data to specific partitions.

        Returns list of (partition_id, regional_data) tuples
        """
        record = json.loads(line.strip())
        region = record.get("region", "unknown")
        user_id = str(record.get("user_id", ""))

        # Determine partition based on strategy
        partition_id = strategy.partition_func(region, user_id, strategy.num_partitions)

        # Extract all relevant data for regional analysis
        timestamp = record.get("timestamp", 0)
        dt = datetime.datetime.fromtimestamp(timestamp)

        regional_data = {
            'region': region,
            'user_id': user_id,
            'timestamp': timestamp,
            'hour_of_day': dt.hour,
            'day_of_week': dt.weekday(),
            'activity_type': record.get("activity_type", "unknown"),
            'engagement_score': record.get("engagement_score", 0),
            'activity_count': 1
        }

        return [(partition_id, regional_data)]

    @staticmethod
    def reduce(key: str, values: List[Dict]) -> Dict:
        """
        Reduce phase: Aggregate regional activity metrics.

        Combines all activity data for a region into comprehensive metrics
        """
        region = key

        # Initialize aggregated metrics
        total_activities = 0
        unique_users = set()
        activity_types = defaultdict(int)
        hourly_patterns = defaultdict(int)
        daily_patterns = defaultdict(int)
        total_engagement = 0
        user_engagement = defaultdict(float)

        # Process all activities for this region
        for activity_data in values:
            total_activities += activity_data['activity_count']
            unique_users.add(activity_data['user_id'])

            activity_type = activity_data['activity_type']
            activity_types[activity_type] += 1

            hour = activity_data['hour_of_day']
            hourly_patterns[hour] += 1

            day = activity_data['day_of_week']
            daily_patterns[day] += 1

            engagement = activity_data['engagement_score']
            total_engagement += engagement
            user_engagement[activity_data['user_id']] += engagement

        # Calculate derived metrics
        unique_user_count = len(unique_users)
        avg_engagement_per_activity = total_engagement / total_activities if total_activities > 0 else 0
        avg_activities_per_user = total_activities / unique_user_count if unique_user_count > 0 else 0

        # Find peak activity patterns
        peak_hour = max(hourly_patterns.items(), key=lambda x: x[1])[0] if hourly_patterns else 0
        peak_day = max(daily_patterns.items(), key=lambda x: x[1])[0] if daily_patterns else 0

        # Calculate activity diversity (number of different activity types)
        activity_diversity = len(activity_types)

        # Find top activity type
        top_activity_type = max(activity_types.items(), key=lambda x: x[1])[0] if activity_types else "unknown"

        return {
            'region': region,
            'total_activities': total_activities,
            'unique_users': unique_user_count,
            'avg_engagement_per_activity': avg_engagement_per_activity,
            'avg_activities_per_user': avg_activities_per_user,
            'activity_types': dict(activity_types),
            'hourly_patterns': dict(hourly_patterns),
            'daily_patterns': dict(daily_patterns),
            'peak_hour': peak_hour,
            'peak_day': peak_day,
            'activity_diversity': activity_diversity,
            'top_activity_type': top_activity_type,
            'total_engagement': total_engagement,
            'user_engagement': dict(user_engagement)
        }

    @staticmethod
    def partition_aware_reduce(partitioned_data: Dict[int, List[Dict]]) -> Dict[int, Dict[str, Dict]]:
        """
        Partition-aware reduce that processes data within each partition.

        Args:
            partitioned_data: Dict mapping partition_id to list of regional data

        Returns:
            Dict mapping partition_id to regional metrics
        """
        result = {}

        for partition_id, regional_data_list in partitioned_data.items():
            # Group by region within this partition
            region_groups = defaultdict(list)
            for regional_data in regional_data_list:
                region = regional_data['region']
                region_groups[region].append(regional_data)

            # Apply reduce function to each region group
            partition_result = {}
            for region, activities in region_groups.items():
                partition_result[region] = RegionalActivityMapReduce.reduce(region, activities)

            result[partition_id] = partition_result

        return result


# Geographic Partitioning Strategies

def hash_partition(region: str, user_id: str, num_partitions: int) -> int:
    """Simple hash partitioning based on region."""
    return hash(region) % num_partitions


def geographic_cluster_partition(region: str, user_id: str, num_partitions: int) -> int:
    """
    Geographic cluster partitioning that groups nearby regions together.

    Groups regions by geographic proximity to optimize data locality
    for cross-regional analysis.
    """
    # Define geographic clusters (in practice, would use actual geographic coordinates)
    region_clusters = {
        # North America
        'north_america_west': ['california', 'oregon', 'washington', 'nevada'],
        'north_america_east': ['new_york', 'florida', 'massachusetts', 'virginia'],
        'north_america_central': ['texas', 'illinois', 'ohio', 'michigan'],

        # Europe
        'europe_west': ['united_kingdom', 'france', 'spain', 'netherlands'],
        'europe_east': ['germany', 'poland', 'czech_republic', 'austria'],
        'europe_north': ['sweden', 'norway', 'denmark', 'finland'],

        # Asia Pacific
        'asia_east': ['japan', 'south_korea', 'china', 'taiwan'],
        'asia_south': ['india', 'pakistan', 'bangladesh', 'sri_lanka'],
        'asia_southeast': ['singapore', 'thailand', 'malaysia', 'indonesia'],

        # Other regions
        'oceania': ['australia', 'new_zealand', 'fiji'],
        'africa': ['south_africa', 'nigeria', 'egypt', 'kenya'],
        'south_america': ['brazil', 'argentina', 'chile', 'colombia']
    }

    # Find which cluster this region belongs to
    region_lower = region.lower().replace(' ', '_')
    cluster_id = 0

    for i, (cluster_name, regions) in enumerate(region_clusters.items()):
        if region_lower in regions:
            cluster_id = i
            break

    # Ensure we don't exceed num_partitions
    return cluster_id % num_partitions


def timezone_aware_partition(region: str, user_id: str, num_partitions: int) -> int:
    """
    Timezone-aware partitioning that groups regions by similar time zones.

    Helps with temporal analysis by keeping regions with similar peak hours together.
    """
    # Define timezone groups (UTC offsets)
    timezone_groups = {
        'utc_minus_8': ['california', 'oregon', 'washington', 'nevada'],
        'utc_minus_5': ['new_york', 'florida', 'massachusetts', 'virginia'],
        'utc_minus_6': ['texas', 'illinois', 'ohio', 'michigan'],
        'utc_0': ['united_kingdom', 'spain', 'portugal'],
        'utc_plus_1': ['france', 'germany', 'netherlands', 'poland'],
        'utc_plus_2': ['finland', 'sweden', 'norway', 'denmark'],
        'utc_plus_5_30': ['india', 'pakistan', 'bangladesh', 'sri_lanka'],
        'utc_plus_8': ['singapore', 'china', 'malaysia', 'philippines'],
        'utc_plus_9': ['japan', 'south_korea'],
        'utc_plus_10': ['australia', 'sydney'],
        'utc_plus_12': ['new_zealand', 'fiji']
    }

    region_lower = region.lower().replace(' ', '_')
    timezone_id = 0

    for i, (tz_name, regions) in enumerate(timezone_groups.items()):
        if region_lower in regions:
            timezone_id = i
            break

    return timezone_id % num_partitions


def activity_density_partition(region: str, user_id: str, num_partitions: int) -> int:
    """
    Activity density-based partitioning that balances load based on regional activity levels.

    High-activity regions are spread across multiple partitions to prevent hotspots.
    """
    # Define activity density tiers (in practice, would be computed from historical data)
    high_activity_regions = [
        'california', 'new_york', 'texas', 'florida', 'united_kingdom',
        'germany', 'france', 'japan', 'south_korea', 'singapore', 'australia'
    ]

    medium_activity_regions = [
        'oregon', 'washington', 'massachusetts', 'illinois', 'spain',
        'netherlands', 'sweden', 'china', 'india', 'canada'
    ]

    region_lower = region.lower().replace(' ', '_')

    if region_lower in high_activity_regions:
        # High activity regions: use salting to spread across multiple partitions
        salt = hash(user_id) % 4  # Spread across 4 partitions
        base_partition = hash(region) % (num_partitions // 2)
        return (base_partition + salt) % num_partitions

    elif region_lower in medium_activity_regions:
        # Medium activity regions: use normal hash partitioning
        return hash(region) % num_partitions

    else:
        # Low activity regions: can be grouped together
        return hash(region) % (num_partitions // 2)


# Analysis and Measurement Functions

def run_regional_experiment():
    """
    Run the regional activity analysis experiment with different partitioning strategies.
    """
    print("🌍 REGIONAL ACTIVITY PATTERN ANALYSIS")
    print("="*50)

    # Get data files
    data_dir = Path(__file__).parent / "social_media_data"
    data_files = list(data_dir.glob("user_activity_*.json"))

    if not data_files:
        print("❌ No data files found. Run data_generator.py first!")
        return

    print(f"📂 Found {len(data_files)} data files")

    # Define partitioning strategies to test
    strategies = [
        PartitioningStrategy("hash", hash_partition, 8),
        PartitioningStrategy("geographic_cluster", geographic_cluster_partition, 8),
        PartitioningStrategy("timezone_aware", timezone_aware_partition, 8),
        PartitioningStrategy("activity_density", activity_density_partition, 8),
    ]

    mapreduce_class = RegionalActivityMapReduce()
    results = {}

    for strategy in strategies:
        print(f"\n🔧 Testing {strategy.name} partitioning...")

        start_time = time.time()

        # Map phase with partitioning
        per_partition_data = defaultdict(list)
        total_records = 0

        for file_path in data_files:
            with open(file_path, "r") as f:
                lines = f.readlines()
                total_records += len(lines)

                for line in lines:
                    if line.strip():
                        per_line_data = mapreduce_class.partition_aware_map(line, strategy)
                        for partition_id, regional_data in per_line_data:
                            per_partition_data[partition_id].append(regional_data)

        # Reduce phase
        reduced_data = mapreduce_class.partition_aware_reduce(per_partition_data)

        processing_time = time.time() - start_time

        # Calculate load balance metrics
        partition_sizes = [len(data) for data in per_partition_data.values()]
        max_partition_size = max(partition_sizes) if partition_sizes else 0
        min_partition_size = min(partition_sizes) if partition_sizes else 0
        avg_partition_size = sum(partition_sizes) / len(partition_sizes) if partition_sizes else 0

        # Calculate load imbalance ratio
        load_imbalance = max_partition_size / avg_partition_size if avg_partition_size > 0 else 0

        # Count total regions processed
        total_regions = set()
        for partition_data in reduced_data.values():
            total_regions.update(partition_data.keys())

        results[strategy.name] = {
            'processing_time': processing_time,
            'load_imbalance': load_imbalance,
            'max_partition_size': max_partition_size,
            'min_partition_size': min_partition_size,
            'avg_partition_size': avg_partition_size,
            'total_regions': len(total_regions),
            'reduced_data': reduced_data
        }

        print(f"   ⏱️  Processing time: {processing_time:.2f}s")
        print(f"   ⚖️  Load imbalance ratio: {load_imbalance:.2f}")
        print(f"   📊 Partition sizes: {min_partition_size} - {max_partition_size} (avg: {avg_partition_size:.1f})")
        print(f"   🌍 Total regions: {len(total_regions)}")

    # Compare strategies
    print(f"\n📊 STRATEGY COMPARISON")
    print("="*30)

    best_balance = min(results.keys(), key=lambda x: results[x]['load_imbalance'])
    fastest = min(results.keys(), key=lambda x: results[x]['processing_time'])

    print(f"🏆 Best load balance: {best_balance} (ratio: {results[best_balance]['load_imbalance']:.2f})")
    print(f"⚡ Fastest processing: {fastest} ({results[fastest]['processing_time']:.2f}s)")

    # Show detailed regional analysis for best strategy
    print(f"\n🔍 DETAILED REGIONAL ANALYSIS ({best_balance})")
    print("="*40)

    best_reduced_data = results[best_balance]['reduced_data']

    # Aggregate all regional data across partitions
    all_regional_data = {}
    for partition_data in best_reduced_data.values():
        for region, metrics in partition_data.items():
            if region in all_regional_data:
                # Merge data from multiple partitions (shouldn't happen with good partitioning)
                print(f"⚠️  Warning: Region {region} found in multiple partitions!")
            all_regional_data[region] = metrics

    # Sort regions by activity level
    sorted_regions = sorted(all_regional_data.items(),
                          key=lambda x: x[1]['total_activities'], reverse=True)

    print(f"📈 Top 10 Most Active Regions:")
    for i, (region, metrics) in enumerate(sorted_regions[:10]):
        print(f"  {i+1:2d}. {region:15s}: {metrics['total_activities']:6,} activities, "
              f"{metrics['unique_users']:4,} users, "
              f"avg engagement: {metrics['avg_engagement_per_activity']:.2f}")

    return results


def analyze_cross_regional_influence():
    """
    Analyze cross-regional influence patterns and data locality requirements.

    Examines how often users from one region interact with content from other regions,
    and measures the efficiency of different partitioning strategies for this analysis.
    """
    print("🌐 CROSS-REGIONAL INFLUENCE ANALYSIS")
    print("="*45)

    # Get data files
    data_dir = Path(__file__).parent / "social_media_data"
    data_files = list(data_dir.glob("user_activity_*.json"))

    if not data_files:
        print("❌ No data files found. Run data_generator.py first!")
        return

    print(f"📂 Analyzing {len(data_files)} data files for cross-regional patterns...")

    # 1. Track user interactions across regions
    print("\n🔗 Cross-Regional Interaction Analysis...")

    # Track user-to-content region interactions
    user_regions = {}  # user_id -> home_region
    cross_regional_interactions = defaultdict(lambda: defaultdict(int))  # home_region -> content_region -> count
    regional_influence_scores = defaultdict(float)  # region -> total influence received from other regions
    total_interactions = 0
    cross_regional_count = 0

    # Parse all data to identify cross-regional interactions
    for file_path in data_files:
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line.strip())
                    user_id = str(record.get("user_id", ""))
                    user_region = record.get("region", "unknown")
                    content_region = record.get("content_region", user_region)  # Assume content region same as user if not specified
                    engagement_score = record.get("engagement_score", 1)

                    # Track user's home region (use most frequent region)
                    if user_id not in user_regions:
                        user_regions[user_id] = user_region

                    total_interactions += 1

                    # Track cross-regional interactions
                    if user_region != content_region:
                        cross_regional_interactions[user_region][content_region] += 1
                        regional_influence_scores[content_region] += engagement_score
                        cross_regional_count += 1

    cross_regional_ratio = cross_regional_count / total_interactions if total_interactions > 0 else 0

    print(f"📊 Total interactions: {total_interactions:,}")
    print(f"🌍 Cross-regional interactions: {cross_regional_count:,} ({cross_regional_ratio*100:.2f}%)")

    # Find most influential cross-regional pairs
    top_cross_regional = []
    for home_region, content_regions in cross_regional_interactions.items():
        for content_region, count in content_regions.items():
            top_cross_regional.append((home_region, content_region, count))

    top_cross_regional.sort(key=lambda x: x[2], reverse=True)

    print(f"\n🔝 Top 10 Cross-Regional Influence Patterns:")
    for i, (home_region, content_region, count) in enumerate(top_cross_regional[:10]):
        percentage = (count / total_interactions * 100) if total_interactions > 0 else 0
        print(f"  {i+1:2d}. {home_region} → {content_region}: {count:4,} interactions ({percentage:.2f}%)")

    # 2. Measure data shuffle requirements for different partitioning strategies
    print(f"\n📦 Data Shuffle Analysis for Cross-Regional Queries...")

    strategies = [
        PartitioningStrategy("hash", hash_partition, 8),
        PartitioningStrategy("geographic_cluster", geographic_cluster_partition, 8),
        PartitioningStrategy("timezone_aware", timezone_aware_partition, 8),
        PartitioningStrategy("activity_density", activity_density_partition, 8),
    ]

    shuffle_analysis = {}

    for strategy in strategies:
        print(f"\n  🔧 Analyzing {strategy.name} partitioning...")

        # Map data to partitions
        region_to_partition = {}
        partition_regions = defaultdict(set)

        # Sample representative regions to determine partition mapping
        sample_regions = list(set(user_regions.values()))[:50]  # Sample regions
        for region in sample_regions:
            partition_id = strategy.partition_func(region, "sample_user", strategy.num_partitions)
            region_to_partition[region] = partition_id
            partition_regions[partition_id].add(region)

        # Calculate shuffle requirements for cross-regional queries
        intra_partition_cross_regional = 0
        inter_partition_cross_regional = 0
        total_shuffle_data = 0

        for home_region, content_regions in cross_regional_interactions.items():
            home_partition = region_to_partition.get(home_region, 0)

            for content_region, count in content_regions.items():
                content_partition = region_to_partition.get(content_region, 0)

                if home_partition == content_partition:
                    intra_partition_cross_regional += count
                else:
                    inter_partition_cross_regional += count
                    total_shuffle_data += count  # Data that needs to be shuffled

        total_cross_regional = intra_partition_cross_regional + inter_partition_cross_regional
        shuffle_ratio = inter_partition_cross_regional / total_cross_regional if total_cross_regional > 0 else 0

        # Calculate partition locality score
        locality_score = intra_partition_cross_regional / total_cross_regional if total_cross_regional > 0 else 0

        shuffle_analysis[strategy.name] = {
            'shuffle_ratio': shuffle_ratio,
            'locality_score': locality_score,
            'intra_partition_interactions': intra_partition_cross_regional,
            'inter_partition_interactions': inter_partition_cross_regional,
            'shuffle_data_volume': total_shuffle_data
        }

        print(f"     🔀 Shuffle ratio: {shuffle_ratio*100:.1f}%")
        print(f"     📍 Locality score: {locality_score*100:.1f}%")
        print(f"     📊 Partitions with regions: {len([p for p in partition_regions.values() if p])}")

    # 3. Calculate cross-regional influence metrics
    print(f"\n🌟 Regional Influence Metrics...")

    # Calculate influence given and received for each region
    influence_given = defaultdict(float)  # How much influence each region gives to others
    influence_received = defaultdict(float)  # How much influence each region receives from others

    for home_region, content_regions in cross_regional_interactions.items():
        for content_region, count in content_regions.items():
            influence_given[home_region] += count
            influence_received[content_region] += count

    # Find most influential regions (high outbound influence)
    top_influencers = sorted(influence_given.items(), key=lambda x: x[1], reverse=True)

    # Find most influenced regions (high inbound influence)
    top_influenced = sorted(influence_received.items(), key=lambda x: x[1], reverse=True)

    print(f"🏆 Top 5 Most Influential Regions (Outbound):")
    for i, (region, influence) in enumerate(top_influencers[:5]):
        print(f"  {i+1}. {region}: {influence:,} outbound interactions")

    print(f"\n🎯 Top 5 Most Influenced Regions (Inbound):")
    for i, (region, influence) in enumerate(top_influenced[:5]):
        print(f"  {i+1}. {region}: {influence:,} inbound interactions")

    # 4. Optimize partitioning for cross-regional queries
    print(f"\n🎯 Cross-Regional Query Optimization Recommendations...")

    best_locality = max(shuffle_analysis.keys(), key=lambda x: shuffle_analysis[x]['locality_score'])
    worst_locality = min(shuffle_analysis.keys(), key=lambda x: shuffle_analysis[x]['locality_score'])

    print(f"✅ Best locality strategy: {best_locality}")
    print(f"   - Locality score: {shuffle_analysis[best_locality]['locality_score']*100:.1f}%")
    print(f"   - Shuffle ratio: {shuffle_analysis[best_locality]['shuffle_ratio']*100:.1f}%")

    print(f"❌ Worst locality strategy: {worst_locality}")
    print(f"   - Locality score: {shuffle_analysis[worst_locality]['locality_score']*100:.1f}%")
    print(f"   - Shuffle ratio: {shuffle_analysis[worst_locality]['shuffle_ratio']*100:.1f}%")

    print(f"\n💡 Optimization Recommendations:")

    if cross_regional_ratio > 0.2:
        print(f"1. HIGH CROSS-REGIONAL ACTIVITY: {cross_regional_ratio*100:.1f}% of interactions are cross-regional")
        print(f"   - Consider replicating popular content across multiple regional partitions")
        print(f"   - Implement caching for frequently accessed cross-regional data")

    avg_shuffle_ratio = sum(data['shuffle_ratio'] for data in shuffle_analysis.values()) / len(shuffle_analysis)
    if avg_shuffle_ratio > 0.5:
        print(f"2. HIGH SHUFFLE OVERHEAD: Average {avg_shuffle_ratio*100:.1f}% data shuffle required")
        print(f"   - Use {best_locality} partitioning for better data locality")
        print(f"   - Consider hierarchical partitioning with regional replication")

    if len(top_influencers) > 0 and top_influencers[0][1] > total_interactions * 0.1:
        top_influencer_region = top_influencers[0][0]
        print(f"3. INFLUENTIAL REGION DETECTED: {top_influencer_region} has high outbound influence")
        print(f"   - Consider dedicated processing resources for {top_influencer_region}")
        print(f"   - Replicate {top_influencer_region} content across multiple partitions")

    print(f"4. GENERAL: Use {best_locality} for cross-regional analytics workloads")
    print(f"5. MONITORING: Track cross-regional interaction patterns for dynamic optimization")

    return {
        'cross_regional_ratio': cross_regional_ratio,
        'top_cross_regional': top_cross_regional[:10],
        'top_influencers': top_influencers[:5],
        'top_influenced': top_influenced[:5],
        'shuffle_analysis': shuffle_analysis,
        'best_locality_strategy': best_locality
    }


def analyze_temporal_patterns():
    """
    Analyze temporal activity patterns across regions and time zones.

    Examines how activity patterns vary by time of day and day of week across
    different geographic regions, accounting for timezone differences.
    """
    print("⏰ TEMPORAL PATTERN ANALYSIS")
    print("="*35)

    # Get data files
    data_dir = Path(__file__).parent / "social_media_data"
    data_files = list(data_dir.glob("user_activity_*.json"))

    if not data_files:
        print("❌ No data files found. Run data_generator.py first!")
        return

    print(f"📂 Analyzing temporal patterns across {len(data_files)} data files...")

    # 1. Analyze activity patterns by hour of day for each region
    print("\n🕐 Hourly Activity Patterns by Region...")

    # Define timezone offsets for major regions
    region_timezones = {
        'california': -8, 'oregon': -8, 'washington': -8, 'nevada': -8,
        'new_york': -5, 'florida': -5, 'massachusetts': -5, 'virginia': -5,
        'texas': -6, 'illinois': -6, 'ohio': -5, 'michigan': -5,
        'united_kingdom': 0, 'france': 1, 'spain': 1, 'netherlands': 1,
        'germany': 1, 'poland': 1, 'sweden': 1, 'norway': 1,
        'japan': 9, 'south_korea': 9, 'china': 8, 'singapore': 8,
        'india': 5.5, 'australia': 10, 'new_zealand': 12
    }

    regional_hourly_patterns = defaultdict(lambda: defaultdict(int))  # region -> hour -> count
    regional_daily_patterns = defaultdict(lambda: defaultdict(int))   # region -> day -> count
    regional_local_hourly = defaultdict(lambda: defaultdict(int))     # region -> local_hour -> count

    total_activities = 0

    # Parse all data to extract temporal patterns
    for file_path in data_files:
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line.strip())
                    region = record.get("region", "unknown").lower().replace(' ', '_')
                    timestamp = record.get("timestamp", 0)

                    dt = datetime.datetime.fromtimestamp(timestamp)
                    utc_hour = dt.hour
                    day_of_week = dt.weekday()

                    # Calculate local hour for this region
                    timezone_offset = region_timezones.get(region, 0)
                    local_hour = (utc_hour + timezone_offset) % 24

                    regional_hourly_patterns[region][utc_hour] += 1
                    regional_daily_patterns[region][day_of_week] += 1
                    regional_local_hourly[region][local_hour] += 1

                    total_activities += 1

    print(f"📊 Total activities analyzed: {total_activities:,}")
    print(f"🌍 Regions found: {len(regional_hourly_patterns)}")

    # 2. Account for timezone differences in activity patterns
    print(f"\n🌐 Timezone-Adjusted Activity Analysis...")

    # Find global vs local peak hours
    global_hourly_totals = defaultdict(int)
    for region_patterns in regional_hourly_patterns.values():
        for hour, count in region_patterns.items():
            global_hourly_totals[hour] += count

    global_peak_hour = max(global_hourly_totals.items(), key=lambda x: x[1])[0]
    global_peak_count = global_hourly_totals[global_peak_hour]

    print(f"🔝 Global peak hour (UTC): {global_peak_hour}:00 ({global_peak_count:,} activities)")

    # Analyze local peaks for each region
    regional_local_peaks = {}
    for region, local_patterns in regional_local_hourly.items():
        if local_patterns:
            local_peak_hour = max(local_patterns.items(), key=lambda x: x[1])[0]
            local_peak_count = local_patterns[local_peak_hour]
            regional_local_peaks[region] = {
                'peak_hour': local_peak_hour,
                'peak_count': local_peak_count,
                'total_activities': sum(local_patterns.values())
            }

    # Sort regions by total activity
    sorted_regions = sorted(regional_local_peaks.items(),
                          key=lambda x: x[1]['total_activities'], reverse=True)

    print(f"\n📈 Top 10 Regions - Local Peak Hours:")
    for i, (region, peak_data) in enumerate(sorted_regions[:10]):
        total = peak_data['total_activities']
        peak_hour = peak_data['peak_hour']
        peak_count = peak_data['peak_count']
        peak_percentage = (peak_count / total * 100) if total > 0 else 0

        print(f"  {i+1:2d}. {region:15s}: Peak at {peak_hour:2d}:00 local "
              f"({peak_count:4,} activities, {peak_percentage:.1f}% of region total)")

    # 3. Identify global vs local activity peaks
    print(f"\n🕰️ Global vs Local Peak Analysis...")

    # Calculate correlation between global and local peaks
    peak_hour_distribution = defaultdict(int)
    regions_with_data = 0

    for region, peak_data in regional_local_peaks.items():
        if peak_data['total_activities'] > 10:  # Only consider regions with sufficient data
            peak_hour_distribution[peak_data['peak_hour']] += 1
            regions_with_data += 1

    if peak_hour_distribution:
        most_common_local_peak = max(peak_hour_distribution.items(), key=lambda x: x[1])
        common_peak_hour = most_common_local_peak[0]
        regions_with_common_peak = most_common_local_peak[1]

        print(f"📊 Most common local peak hour: {common_peak_hour}:00 "
              f"({regions_with_common_peak}/{regions_with_data} regions)")

        # Check if regions follow global patterns or have distinct local patterns
        synchronized_regions = 0
        asynchronous_regions = 0

        for region, peak_data in regional_local_peaks.items():
            local_peak = peak_data['peak_hour']
            # Consider synchronized if within 2 hours of most common peak
            if abs(local_peak - common_peak_hour) <= 2 or abs(local_peak - common_peak_hour) >= 22:
                synchronized_regions += 1
            else:
                asynchronous_regions += 1

        sync_percentage = (synchronized_regions / len(regional_local_peaks) * 100) if regional_local_peaks else 0

        print(f"🔄 Synchronized regions: {synchronized_regions} ({sync_percentage:.1f}%)")
        print(f"🌍 Asynchronous regions: {asynchronous_regions}")

    # 4. Measure temporal skew impact on partitioning strategies
    print(f"\n⚖️ Temporal Skew Impact on Partitioning...")

    strategies = [
        PartitioningStrategy("hash", hash_partition, 8),
        PartitioningStrategy("geographic_cluster", geographic_cluster_partition, 8),
        PartitioningStrategy("timezone_aware", timezone_aware_partition, 8),
        PartitioningStrategy("activity_density", activity_density_partition, 8),
    ]

    temporal_analysis = {}

    for strategy in strategies:
        print(f"\n  🔧 Analyzing temporal skew for {strategy.name}...")

        # Simulate hourly load distribution across partitions
        hourly_partition_loads = defaultdict(lambda: defaultdict(int))  # hour -> partition -> load

        # Map regions to partitions
        region_to_partition = {}
        for region in regional_hourly_patterns.keys():
            partition_id = strategy.partition_func(region, "sample_user", strategy.num_partitions)
            region_to_partition[region] = partition_id

        # Calculate hourly load per partition
        for region, hourly_pattern in regional_hourly_patterns.items():
            partition_id = region_to_partition.get(region, 0)
            for hour, count in hourly_pattern.items():
                hourly_partition_loads[hour][partition_id] += count

        # Calculate temporal skew metrics
        max_hourly_imbalance = 0
        avg_hourly_imbalance = 0
        peak_hour_max_load = 0
        peak_hour_min_load = float('inf')

        for hour, partition_loads in hourly_partition_loads.items():
            if partition_loads:
                max_load = max(partition_loads.values())
                min_load = min(partition_loads.values())
                avg_load = sum(partition_loads.values()) / len(partition_loads)

                hourly_imbalance = max_load / avg_load if avg_load > 0 else 0
                max_hourly_imbalance = max(max_hourly_imbalance, hourly_imbalance)
                avg_hourly_imbalance += hourly_imbalance

                # Track peak hour performance
                if hour == global_peak_hour:
                    peak_hour_max_load = max_load
                    peak_hour_min_load = min_load

        avg_hourly_imbalance /= len(hourly_partition_loads) if hourly_partition_loads else 1
        peak_hour_imbalance = peak_hour_max_load / peak_hour_min_load if peak_hour_min_load > 0 else 0

        temporal_analysis[strategy.name] = {
            'max_hourly_imbalance': max_hourly_imbalance,
            'avg_hourly_imbalance': avg_hourly_imbalance,
            'peak_hour_imbalance': peak_hour_imbalance,
            'peak_hour_max_load': peak_hour_max_load,
            'peak_hour_min_load': peak_hour_min_load
        }

        print(f"     ⚖️ Avg hourly imbalance: {avg_hourly_imbalance:.2f}")
        print(f"     🔝 Peak hour imbalance: {peak_hour_imbalance:.2f}")
        print(f"     📊 Max hourly imbalance: {max_hourly_imbalance:.2f}")

    # Recommendations based on temporal analysis
    print(f"\n🎯 Temporal Partitioning Recommendations...")

    best_temporal = min(temporal_analysis.keys(),
                       key=lambda x: temporal_analysis[x]['avg_hourly_imbalance'])
    worst_temporal = max(temporal_analysis.keys(),
                        key=lambda x: temporal_analysis[x]['avg_hourly_imbalance'])

    print(f"✅ Best temporal balance: {best_temporal}")
    print(f"   - Avg hourly imbalance: {temporal_analysis[best_temporal]['avg_hourly_imbalance']:.2f}")
    print(f"   - Peak hour imbalance: {temporal_analysis[best_temporal]['peak_hour_imbalance']:.2f}")

    print(f"❌ Worst temporal balance: {worst_temporal}")
    print(f"   - Avg hourly imbalance: {temporal_analysis[worst_temporal]['avg_hourly_imbalance']:.2f}")
    print(f"   - Peak hour imbalance: {temporal_analysis[worst_temporal]['peak_hour_imbalance']:.2f}")

    print(f"\n💡 Temporal Optimization Recommendations:")

    avg_peak_imbalance = sum(data['peak_hour_imbalance'] for data in temporal_analysis.values()) / len(temporal_analysis)
    if avg_peak_imbalance > 3.0:
        print(f"1. HIGH PEAK HOUR SKEW: Average peak imbalance is {avg_peak_imbalance:.2f}")
        print(f"   - Implement dynamic load balancing during peak hours ({global_peak_hour}:00 UTC)")
        print(f"   - Use temporal partitioning with peak hour redistribution")

    if sync_percentage < 60:  # Most regions have different peak hours
        print(f"2. ASYNCHRONOUS ACTIVITY: Only {sync_percentage:.1f}% of regions are synchronized")
        print(f"   - Leverage timezone-aware partitioning for better load distribution")
        print(f"   - Implement follow-the-sun processing architecture")

    if max(data['max_hourly_imbalance'] for data in temporal_analysis.values()) > 5.0:
        print(f"3. EXTREME TEMPORAL SKEW: Some hours show >5x load imbalance")
        print(f"   - Implement adaptive partitioning that changes based on time of day")
        print(f"   - Use elastic scaling during peak hours")

    print(f"4. GENERAL: Use {best_temporal} partitioning for time-sensitive workloads")
    print(f"5. MONITORING: Track hourly load patterns for dynamic optimization")

    return {
        'global_peak_hour': global_peak_hour,
        'regional_local_peaks': dict(regional_local_peaks),
        'synchronized_percentage': sync_percentage,
        'temporal_analysis': temporal_analysis,
        'best_temporal_strategy': best_temporal
    }


def optimize_geographic_partitioning():
    """
    Optimize geographic partitioning strategies based on data characteristics.

    Uses regional activity metrics and cross-regional interaction patterns
    to recommend optimal partitioning strategies.
    """
    print("🎯 GEOGRAPHIC PARTITIONING OPTIMIZATION")
    print("="*45)

    # Get data files
    data_dir = Path(__file__).parent / "social_media_data"
    data_files = list(data_dir.glob("user_activity_*.json"))

    if not data_files:
        print("❌ No data files found. Run data_generator.py first!")
        return

    print(f"📂 Analyzing {len(data_files)} data files for optimization opportunities...")

    # 1. Analyze current geographic data distribution
    print("\n📊 Geographic Data Distribution Analysis...")

    # Collect regional activity metrics
    regional_metrics = defaultdict(lambda: {
        'total_activities': 0,
        'unique_users': set(),
        'total_engagement': 0,
        'activity_types': defaultdict(int),
        'hourly_distribution': defaultdict(int)
    })

    total_global_activities = 0

    # Parse all data to get comprehensive regional metrics
    for file_path in data_files:
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line.strip())
                    region = record.get("region", "unknown")
                    user_id = str(record.get("user_id", ""))
                    engagement = record.get("engagement_score", 0)
                    activity_type = record.get("activity_type", "unknown")
                    timestamp = record.get("timestamp", 0)

                    dt = datetime.datetime.fromtimestamp(timestamp)
                    hour = dt.hour

                    regional_metrics[region]['total_activities'] += 1
                    regional_metrics[region]['unique_users'].add(user_id)
                    regional_metrics[region]['total_engagement'] += engagement
                    regional_metrics[region]['activity_types'][activity_type] += 1
                    regional_metrics[region]['hourly_distribution'][hour] += 1

                    total_global_activities += 1

    # Convert sets to counts for easier processing
    for region in regional_metrics:
        regional_metrics[region]['unique_users'] = len(regional_metrics[region]['unique_users'])

    # Calculate derived metrics
    for region, metrics in regional_metrics.items():
        activities = metrics['total_activities']
        metrics['activity_density'] = activities / total_global_activities if total_global_activities > 0 else 0
        metrics['avg_engagement'] = metrics['total_engagement'] / activities if activities > 0 else 0
        metrics['users_per_activity'] = metrics['unique_users'] / activities if activities > 0 else 0

    print(f"🌍 Total regions: {len(regional_metrics)}")
    print(f"📊 Total activities: {total_global_activities:,}")

    # Sort regions by activity level
    sorted_regions = sorted(regional_metrics.items(),
                          key=lambda x: x[1]['total_activities'], reverse=True)

    print(f"\n📈 Regional Activity Distribution:")
    for i, (region, metrics) in enumerate(sorted_regions[:10]):
        density = metrics['activity_density'] * 100
        print(f"  {i+1:2d}. {region:20s}: {metrics['total_activities']:6,} activities "
              f"({density:5.2f}%), {metrics['unique_users']:4,} users, "
              f"avg engagement: {metrics['avg_engagement']:.2f}")

    # 2. Identify hotspot regions and load imbalances
    print(f"\n🔥 Hotspot Region Identification...")

    # Calculate activity thresholds
    avg_regional_activity = total_global_activities / len(regional_metrics) if regional_metrics else 0
    high_activity_threshold = avg_regional_activity * 3  # 3x average
    hotspot_threshold = avg_regional_activity * 5        # 5x average

    hotspot_regions = []
    high_activity_regions = []
    low_activity_regions = []

    for region, metrics in regional_metrics.items():
        activities = metrics['total_activities']

        if activities >= hotspot_threshold:
            hotspot_regions.append((region, metrics))
        elif activities >= high_activity_threshold:
            high_activity_regions.append((region, metrics))
        else:
            low_activity_regions.append((region, metrics))

    print(f"🔥 Hotspot regions (>{hotspot_threshold:.0f} activities): {len(hotspot_regions)}")
    for region, metrics in sorted(hotspot_regions, key=lambda x: x[1]['total_activities'], reverse=True):
        ratio = metrics['total_activities'] / avg_regional_activity
        print(f"   {region}: {metrics['total_activities']:,} activities ({ratio:.1f}x average)")

    print(f"⚡ High activity regions (>{high_activity_threshold:.0f} activities): {len(high_activity_regions)}")
    print(f"🟢 Normal activity regions: {len(low_activity_regions)}")

    # Calculate geographic skew metrics
    max_regional_activity = max(metrics['total_activities'] for metrics in regional_metrics.values())
    min_regional_activity = min(metrics['total_activities'] for metrics in regional_metrics.values())
    geographic_skew_ratio = max_regional_activity / avg_regional_activity if avg_regional_activity > 0 else 0

    print(f"\n⚖️ Geographic Load Balance Analysis:")
    print(f"   Max regional activity: {max_regional_activity:,}")
    print(f"   Min regional activity: {min_regional_activity:,}")
    print(f"   Average regional activity: {avg_regional_activity:.1f}")
    print(f"   Geographic skew ratio: {geographic_skew_ratio:.2f}")

    # 3. Recommend partitioning strategy adjustments
    print(f"\n🔧 Partitioning Strategy Optimization...")

    strategies = [
        PartitioningStrategy("hash", hash_partition, 8),
        PartitioningStrategy("geographic_cluster", geographic_cluster_partition, 8),
        PartitioningStrategy("timezone_aware", timezone_aware_partition, 8),
        PartitioningStrategy("activity_density", activity_density_partition, 8),
    ]

    strategy_performance = {}

    for strategy in strategies:
        print(f"\n  📊 Evaluating {strategy.name} strategy...")

        # Simulate partitioning with current data
        partition_loads = defaultdict(int)
        partition_regions = defaultdict(list)
        hotspot_distribution = defaultdict(int)

        for region, metrics in regional_metrics.items():
            partition_id = strategy.partition_func(region, "sample_user", strategy.num_partitions)
            partition_loads[partition_id] += metrics['total_activities']
            partition_regions[partition_id].append(region)

            # Track hotspot distribution
            if metrics['total_activities'] >= hotspot_threshold:
                hotspot_distribution[partition_id] += 1

        # Calculate balance metrics
        if partition_loads:
            max_partition_load = max(partition_loads.values())
            min_partition_load = min(partition_loads.values())
            avg_partition_load = sum(partition_loads.values()) / len(partition_loads)

            load_imbalance_ratio = max_partition_load / avg_partition_load if avg_partition_load > 0 else 0
            min_max_ratio = min_partition_load / max_partition_load if max_partition_load > 0 else 0

            # Count partitions with hotspots
            partitions_with_hotspots = len([p for p in hotspot_distribution.values() if p > 0])
            max_hotspots_per_partition = max(hotspot_distribution.values()) if hotspot_distribution else 0

            strategy_performance[strategy.name] = {
                'load_imbalance_ratio': load_imbalance_ratio,
                'min_max_ratio': min_max_ratio,
                'partitions_with_hotspots': partitions_with_hotspots,
                'max_hotspots_per_partition': max_hotspots_per_partition,
                'max_partition_load': max_partition_load,
                'min_partition_load': min_partition_load,
                'avg_partition_load': avg_partition_load
            }

            print(f"     ⚖️ Load imbalance ratio: {load_imbalance_ratio:.2f}")
            print(f"     📊 Min/Max load ratio: {min_max_ratio:.2f}")
            print(f"     🔥 Partitions with hotspots: {partitions_with_hotspots}/{strategy.num_partitions}")
            print(f"     📈 Max hotspots per partition: {max_hotspots_per_partition}")

    # Find best and worst strategies
    best_balance = min(strategy_performance.keys(),
                      key=lambda x: strategy_performance[x]['load_imbalance_ratio'])
    worst_balance = max(strategy_performance.keys(),
                       key=lambda x: strategy_performance[x]['load_imbalance_ratio'])

    best_hotspot = min(strategy_performance.keys(),
                      key=lambda x: strategy_performance[x]['max_hotspots_per_partition'])

    print(f"\n🏆 Strategy Performance Summary:")
    print(f"✅ Best load balance: {best_balance} (ratio: {strategy_performance[best_balance]['load_imbalance_ratio']:.2f})")
    print(f"❌ Worst load balance: {worst_balance} (ratio: {strategy_performance[worst_balance]['load_imbalance_ratio']:.2f})")
    print(f"🔥 Best hotspot distribution: {best_hotspot} (max {strategy_performance[best_hotspot]['max_hotspots_per_partition']} hotspots/partition)")

    # 4. Provide data locality optimization suggestions
    print(f"\n💡 Geographic Partitioning Optimization Recommendations:")

    if geographic_skew_ratio > 10:
        print(f"1. EXTREME GEOGRAPHIC SKEW: {geographic_skew_ratio:.1f}x imbalance detected")
        print(f"   - Implement salting for hotspot regions: {[r[0] for r in hotspot_regions]}")
        print(f"   - Use activity-density partitioning with dynamic load balancing")
        print(f"   - Consider dedicated processing pools for hotspot regions")

    if len(hotspot_regions) > 0:
        hotspot_names = [r[0] for r in hotspot_regions[:3]]
        print(f"2. HOTSPOT MANAGEMENT: {len(hotspot_regions)} hotspot regions identified")
        print(f"   - Primary hotspots: {', '.join(hotspot_names)}")
        print(f"   - Replicate hotspot data across multiple partitions")
        print(f"   - Use {best_hotspot} strategy for better hotspot distribution")

    avg_imbalance = sum(perf['load_imbalance_ratio'] for perf in strategy_performance.values()) / len(strategy_performance)
    if avg_imbalance > 3.0:
        print(f"3. HIGH LOAD IMBALANCE: Average imbalance ratio is {avg_imbalance:.2f}")
        print(f"   - Switch from worst strategy ({worst_balance}) to best strategy ({best_balance})")
        print(f"   - Implement dynamic partition splitting for overloaded partitions")

    max_hotspots = max(perf['max_hotspots_per_partition'] for perf in strategy_performance.values())
    if max_hotspots > 2:
        print(f"4. HOTSPOT CONCENTRATION: Up to {max_hotspots} hotspots in single partition")
        print(f"   - Use composite partitioning: geographic clustering + activity-based salting")
        print(f"   - Implement hotspot detection and automatic redistribution")

    # Geographic locality recommendations
    if len(regional_metrics) > strategy.num_partitions * 2:
        print(f"5. REGION FRAGMENTATION: {len(regional_metrics)} regions across {strategy.num_partitions} partitions")
        print(f"   - Use geographic clustering to group nearby regions")
        print(f"   - Consider hierarchical partitioning by continent/country/region")

    print(f"6. OPTIMAL CONFIGURATION:")
    print(f"   - Primary strategy: {best_balance} for overall load balance")
    print(f"   - Hotspot handling: {best_hotspot} for hotspot distribution")
    print(f"   - Monitor regional activity patterns for adaptive optimization")

    # Data locality optimization
    print(f"\n🗺️ Data Locality Optimization:")

    # Estimate cross-partition access for best strategy
    regions_per_partition = len(regional_metrics) / strategy.num_partitions
    if regions_per_partition > 3:  # Multiple regions per partition
        print(f"   📍 {best_balance}: ~{regions_per_partition:.1f} regions per partition")
        print(f"   - Good locality for intra-regional queries")
        print(f"   - May require cross-partition access for multi-regional analysis")

    print(f"   🎯 Recommendations:")
    print(f"   - Cache frequently accessed cross-regional data")
    print(f"   - Use region-aware query planning")
    print(f"   - Implement regional data replication for global queries")

    return {
        'total_regions': len(regional_metrics),
        'hotspot_regions': len(hotspot_regions),
        'geographic_skew_ratio': geographic_skew_ratio,
        'best_balance_strategy': best_balance,
        'best_hotspot_strategy': best_hotspot,
        'strategy_performance': strategy_performance,
        'optimization_priority': 'high' if geographic_skew_ratio > 5 else 'medium' if geographic_skew_ratio > 3 else 'low'
    }


if __name__ == "__main__":
    print("🎯 CHALLENGE 4: Regional Activity Patterns with Geographic Partitioning")
    print("="*70)
    print()
    print("Your mission:")
    print("1. Complete the TODOs in this file")
    print("2. Implement different geographic partitioning strategies")
    print("3. Handle regional data skew and optimize data locality")
    print("4. Measure and compare partitioning effectiveness for regional analysis")
    print()
    print("Key concepts to implement:")
    print("- Geographic cluster partitioning")
    print("- Timezone-aware partitioning")
    print("- Activity density-based load balancing")
    print("- Cross-regional influence analysis")
    print("- Temporal pattern analysis across time zones")
    print("- Data locality optimization for regional queries")
    print()
    print("Run this file when you've implemented the TODOs!")
    print()

    # Uncomment when ready to test
    # run_regional_experiment()
    # analyze_cross_regional_influence()
    # analyze_temporal_patterns()
    # optimize_geographic_partitioning()
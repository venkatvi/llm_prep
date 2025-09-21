"""
Partition Load Analysis and Visualization Framework

This module provides tools to measure and visualize the effectiveness of
different partitioning strategies for handling data skew in MapReduce.
"""

import json
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Callable
import matplotlib.pyplot as plt
import numpy as np


class PartitionAnalyzer:
    """Analyzes partition load distribution and effectiveness."""

    def __init__(self, num_partitions: int = 8):
        self.num_partitions = num_partitions
        self.partition_strategies = {
            'hash': self._hash_partition,
            'range': self._range_partition,
            'user_tier': self._user_tier_partition,
            'custom_balanced': self._custom_balanced_partition
        }

    def _hash_partition(self, key: str, data: Dict = None) -> int:
        """Simple hash-based partitioning."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % self.num_partitions

    def _range_partition(self, key: str, data: Dict = None) -> int:
        """Range-based partitioning (for sortable keys)."""
        if key.isdigit():
            key_val = int(key)
            ranges_per_partition = 10000 // self.num_partitions
            return min(key_val // ranges_per_partition, self.num_partitions - 1)
        else:
            # Fallback to hash for non-numeric keys
            return self._hash_partition(key, data)

    def _user_tier_partition(self, key: str, data: Dict = None) -> int:
        """Custom partitioning based on user tier to handle power users."""
        if data and 'user_tier' in data:
            tier = data['user_tier']
            # Distribute power users across more partitions
            if tier == 'power_user':
                # Use multiple partitions for power users
                user_id = int(key) if key.isdigit() else hash(key)
                return user_id % min(4, self.num_partitions)  # Use first 4 partitions
            elif tier == 'active_user':
                return 4 % self.num_partitions  # Dedicated partition
            elif tier == 'regular_user':
                return 5 % self.num_partitions
            else:  # lurker
                return (6 + hash(key)) % max(1, self.num_partitions - 6)
        return self._hash_partition(key, data)

    def _custom_balanced_partition(self, key: str, data: Dict = None) -> int:
        """Smart partitioning that tries to balance load."""
        # This would use runtime statistics in a real implementation
        # For demo, we'll use a combination of strategies
        key_hash = hash(key)

        # Try to detect hot keys and spread them
        if data and ('is_viral' in data and data['is_viral']):
            # Spread viral content across all partitions
            return key_hash % self.num_partitions

        return (key_hash // 7) % self.num_partitions  # Different hash function

    def analyze_partitioning(self, data_file: str, key_extractor: Callable,
                           strategy: str = 'hash') -> Dict[str, Any]:
        """Analyze how data distributes across partitions with given strategy."""
        if strategy not in self.partition_strategies:
            raise ValueError(f"Unknown strategy: {strategy}")

        partition_func = self.partition_strategies[strategy]
        partition_loads = defaultdict(int)
        partition_keys = defaultdict(set)
        key_frequencies = defaultdict(int)
        hot_keys = []

        # Read and analyze data
        with open(data_file, 'r') as f:
            for line in f:
                record = json.loads(line.strip())
                key = key_extractor(record)
                partition_id = partition_func(str(key), record)

                partition_loads[partition_id] += 1
                partition_keys[partition_id].add(key)
                key_frequencies[key] += 1

        # Identify hot keys (top 5% by frequency)
        sorted_keys = sorted(key_frequencies.items(), key=lambda x: x[1], reverse=True)
        hot_key_threshold = len(sorted_keys) * 0.05
        hot_keys = [k for k, freq in sorted_keys[:int(hot_key_threshold)]]

        # Calculate metrics
        loads = list(partition_loads.values())
        total_records = sum(loads)

        metrics = {
            'strategy': strategy,
            'total_records': total_records,
            'partition_loads': dict(partition_loads),
            'load_balance_ratio': max(loads) / (total_records / self.num_partitions) if loads else 0,
            'load_variance': np.var(loads) if loads else 0,
            'load_std': np.std(loads) if loads else 0,
            'hot_keys_count': len(hot_keys),
            'hot_keys': hot_keys[:10],  # Top 10 hot keys
            'unique_keys_per_partition': {pid: len(keys) for pid, keys in partition_keys.items()},
            'gini_coefficient': self._calculate_gini(loads)
        }

        return metrics

    def _calculate_gini(self, values: List[int]) -> float:
        """Calculate Gini coefficient to measure inequality in load distribution."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        return (n + 1 - 2 * sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0

    def compare_strategies(self, data_file: str, key_extractor: Callable) -> Dict[str, Dict]:
        """Compare all partitioning strategies on the same dataset."""
        results = {}
        for strategy in self.partition_strategies.keys():
            print(f"Analyzing {strategy} partitioning...")
            results[strategy] = self.analyze_partitioning(data_file, key_extractor, strategy)
        return results

    def visualize_partition_loads(self, comparison_results: Dict[str, Dict],
                                title: str = "Partition Load Distribution"):
        """Create visualization comparing partition load distributions."""
        strategies = list(comparison_results.keys())
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)

        for idx, strategy in enumerate(strategies[:4]):  # Max 4 strategies
            ax = axes[idx // 2, idx % 2]

            result = comparison_results[strategy]
            partition_loads = result['partition_loads']

            partitions = list(range(self.num_partitions))
            loads = [partition_loads.get(p, 0) for p in partitions]

            # Bar plot
            bars = ax.bar(partitions, loads, alpha=0.7)
            ax.set_title(f'{strategy.title()} Partitioning')
            ax.set_xlabel('Partition ID')
            ax.set_ylabel('Number of Records')

            # Color bars by load (red for overloaded)
            max_load = max(loads) if loads else 0
            avg_load = result['total_records'] / self.num_partitions if result['total_records'] > 0 else 0

            for bar, load in zip(bars, loads):
                if load > avg_load * 1.5:  # Overloaded partition
                    bar.set_color('red')
                elif load < avg_load * 0.5:  # Underloaded partition
                    bar.set_color('orange')
                else:
                    bar.set_color('green')

            # Add metrics text
            metrics_text = f"Balance Ratio: {result['load_balance_ratio']:.2f}\n"
            metrics_text += f"Gini Coeff: {result['gini_coefficient']:.3f}\n"
            metrics_text += f"Hot Keys: {result['hot_keys_count']}"

            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

        plt.tight_layout()
        plt.savefig('partition_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def print_detailed_analysis(self, comparison_results: Dict[str, Dict]):
        """Print detailed analysis of partitioning strategies."""
        print("\n" + "="*80)
        print("DETAILED PARTITIONING ANALYSIS")
        print("="*80)

        for strategy, result in comparison_results.items():
            print(f"\n🔹 {strategy.upper()} PARTITIONING")
            print("-" * 40)
            print(f"Total Records: {result['total_records']:,}")
            print(f"Load Balance Ratio: {result['load_balance_ratio']:.2f}")
            print(f"Load Standard Deviation: {result['load_std']:.2f}")
            print(f"Gini Coefficient: {result['gini_coefficient']:.3f}")
            print(f"Hot Keys Detected: {result['hot_keys_count']}")

            if result['hot_keys']:
                print(f"Top Hot Keys: {result['hot_keys']}")

            # Show partition distribution
            loads = list(result['partition_loads'].values())
            if loads:
                print(f"Min Load: {min(loads)}")
                print(f"Max Load: {max(loads)}")
                print(f"Avg Load: {sum(loads)/len(loads):.1f}")

        # Rank strategies
        print(f"\n📊 STRATEGY RANKING (by load balance):")
        print("-" * 40)
        ranked = sorted(comparison_results.items(), key=lambda x: x[1]['load_balance_ratio'])
        for i, (strategy, result) in enumerate(ranked, 1):
            print(f"{i}. {strategy}: {result['load_balance_ratio']:.2f}")


def analyze_user_activity_skew():
    """Analyze user activity data for partitioning effectiveness."""
    print("🔍 ANALYZING USER ACTIVITY PARTITIONING")

    analyzer = PartitionAnalyzer(num_partitions=8)

    # Key extractor for user activity
    def extract_user_id(record):
        return record['user_id']

    # Find a user activity file
    data_files = list(Path("social_media_data").glob("user_activity_*.jsonl"))
    if not data_files:
        print("❌ No user activity files found. Run data_generator.py first.")
        return

    # Analyze first file
    results = analyzer.compare_strategies(str(data_files[0]), extract_user_id)
    analyzer.print_detailed_analysis(results)
    analyzer.visualize_partition_loads(results, "User Activity Partitioning Analysis")

    return results


def analyze_content_engagement_skew():
    """Analyze content engagement data for partitioning effectiveness."""
    print("\n🔍 ANALYZING CONTENT ENGAGEMENT PARTITIONING")

    analyzer = PartitionAnalyzer(num_partitions=6)

    # Key extractor for content engagement
    def extract_content_id(record):
        return record['content_id']

    # Find a content engagement file
    data_files = list(Path("social_media_data").glob("content_engagement_*.jsonl"))
    if not data_files:
        print("❌ No content engagement files found. Run data_generator.py first.")
        return

    # Analyze first file
    results = analyzer.compare_strategies(str(data_files[0]), extract_content_id)
    analyzer.print_detailed_analysis(results)
    analyzer.visualize_partition_loads(results, "Content Engagement Partitioning Analysis")

    return results


if __name__ == "__main__":
    print("🚀 PARTITIONING ANALYSIS FRAMEWORK")
    print("="*50)

    try:
        user_results = analyze_user_activity_skew()
        content_results = analyze_content_engagement_skew()

        print("\n✅ Analysis complete! Check 'partition_analysis.png' for visualizations.")

    except ImportError as e:
        if "matplotlib" in str(e):
            print("⚠️  matplotlib not available, running analysis without visualization...")

            # Run without matplotlib
            analyzer = PartitionAnalyzer()
            data_files = list(Path("social_media_data").glob("user_activity_*.jsonl"))
            if data_files:
                results = analyzer.compare_strategies(str(data_files[0]), lambda r: r['user_id'])
                analyzer.print_detailed_analysis(results)
        else:
            raise
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("Make sure to run 'python data_generator.py' first to generate the datasets.")
"""
Social Media Data Generator for MapReduce Partitioning and Skew Analysis

Generates realistic skewed datasets that mirror real-world social media patterns:
- Power users (Pareto distribution)
- Viral content (few posts get massive engagement)
- Geographic clustering
- Temporal patterns

This data will expose the need for sophisticated partitioning strategies.
"""

import random
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np


class SocialMediaDataGenerator:
    """Generates realistic social media data with controlled skew patterns."""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

        # Configure skew parameters
        self.total_users = 1000
        self.total_content = 5000
        self.power_user_ratio = 0.01  # 1% of users are power users
        self.viral_content_ratio = 0.005  # 0.5% of content goes viral

        # Generate user tiers with Pareto distribution
        self.users = self._generate_users()
        self.content_ids = list(range(1, self.total_content + 1))
        self.hashtags = self._generate_hashtags()

    def _generate_users(self) -> Dict[int, Dict]:
        """Generate users with power-law activity distribution."""
        users = {}

        # Generate activity levels using Pareto distribution
        # Shape parameter controls skew (lower = more skewed)
        activity_levels = np.random.pareto(0.5, self.total_users) + 1
        activity_levels = np.sort(activity_levels)[::-1]  # Sort descending

        for user_id in range(1, self.total_users + 1):
            activity_multiplier = activity_levels[user_id - 1]

            # Classify user tiers
            if user_id <= self.total_users * self.power_user_ratio:
                tier = "power_user"
                base_activity = 1000
            elif user_id <= self.total_users * 0.1:
                tier = "active_user"
                base_activity = 100
            elif user_id <= self.total_users * 0.4:
                tier = "regular_user"
                base_activity = 20
            else:
                tier = "lurker"
                base_activity = 2

            users[user_id] = {
                "tier": tier,
                "activity_level": int(base_activity * activity_multiplier),
                "region": random.choice(["NA", "EU", "APAC", "LATAM", "MEA"])
            }

        return users

    def _generate_hashtags(self) -> List[str]:
        """Generate hashtags with realistic distribution."""
        # Some hashtags will be much more popular
        trending_tags = ["#viral", "#trending", "#breaking", "#news", "#funny"]
        regular_tags = [f"#topic{i}" for i in range(100)]
        niche_tags = [f"#niche{i}" for i in range(1000)]

        return trending_tags + regular_tags + niche_tags

    def generate_user_activity_logs(self, num_records: int, output_dir: str):
        """Generate user activity logs with realistic skew patterns."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        activities = []
        action_types = ["post", "like", "share", "comment", "follow"]

        print(f"Generating {num_records} user activity records...")

        for _ in range(num_records):
            # Select user with probability proportional to their activity level
            user_weights = [self.users[uid]["activity_level"] for uid in self.users.keys()]
            user_id = random.choices(list(self.users.keys()), weights=user_weights, k=1)[0]

            # Generate timestamp with temporal clustering
            base_time = datetime.now() - timedelta(days=30)
            # Add clustering around peak hours (12pm, 6pm, 9pm)
            hour_bias = random.choices([12, 18, 21, random.randint(0, 23)],
                                     weights=[3, 3, 2, 1], k=1)[0]
            timestamp = base_time + timedelta(
                days=random.randint(0, 29),
                hours=hour_bias,
                minutes=random.randint(0, 59)
            )

            activity = {
                "user_id": user_id,
                "action_type": random.choice(action_types),
                "timestamp": timestamp.isoformat(),
                "content_id": random.choice(self.content_ids),
                "user_tier": self.users[user_id]["tier"],
                "region": self.users[user_id]["region"]
            }
            activities.append(activity)

        # Write to multiple files to simulate distributed data
        files_count = 8
        chunk_size = len(activities) // files_count

        for i in range(files_count):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < files_count - 1 else len(activities)
            chunk = activities[start_idx:end_idx]

            file_path = output_path / f"user_activity_{i+1}.jsonl"
            with open(file_path, 'w') as f:
                for record in chunk:
                    f.write(json.dumps(record) + '\n')

        print(f"Generated activity logs in {files_count} files")
        self._print_skew_analysis(activities)

    def generate_content_engagement(self, num_records: int, output_dir: str):
        """Generate content engagement data with viral content skew."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        engagements = []
        engagement_types = ["view", "like", "share", "comment", "save"]

        print(f"Generating {num_records} content engagement records...")

        # Create viral content list (small percentage gets massive engagement)
        viral_content_count = int(self.total_content * self.viral_content_ratio)
        viral_content_ids = random.sample(self.content_ids, viral_content_count)

        for _ in range(num_records):
            # 70% chance of engaging with viral content, 30% with regular content
            if random.random() < 0.7 and viral_content_ids:
                content_id = random.choice(viral_content_ids)
                engagement_multiplier = random.uniform(10, 100)  # Viral boost
            else:
                content_id = random.choice(self.content_ids)
                engagement_multiplier = 1

            # Select user (power users engage more)
            user_weights = [self.users[uid]["activity_level"] for uid in self.users.keys()]
            user_id = random.choices(list(self.users.keys()), weights=user_weights, k=1)[0]

            engagement = {
                "content_id": content_id,
                "user_id": user_id,
                "engagement_type": random.choice(engagement_types),
                "value": int(random.uniform(1, 5) * engagement_multiplier),
                "hashtags": random.sample(self.hashtags, random.randint(0, 3)),
                "is_viral": content_id in viral_content_ids
            }
            engagements.append(engagement)

        # Write to files
        files_count = 6
        chunk_size = len(engagements) // files_count

        for i in range(files_count):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < files_count - 1 else len(engagements)
            chunk = engagements[start_idx:end_idx]

            file_path = output_path / f"content_engagement_{i+1}.jsonl"
            with open(file_path, 'w') as f:
                for record in chunk:
                    f.write(json.dumps(record) + '\n')

        print(f"Generated engagement data in {files_count} files")
        self._print_content_skew_analysis(engagements, viral_content_ids)

    def _print_skew_analysis(self, activities: List[Dict]):
        """Print analysis of data skew in user activities."""
        user_activity_counts = {}
        for activity in activities:
            user_id = activity["user_id"]
            user_activity_counts[user_id] = user_activity_counts.get(user_id, 0) + 1

        sorted_counts = sorted(user_activity_counts.values(), reverse=True)
        total_activities = len(activities)

        # Calculate Pareto statistics
        top_1_percent = int(len(sorted_counts) * 0.01)
        top_10_percent = int(len(sorted_counts) * 0.1)

        top_1_percent_activity = sum(sorted_counts[:top_1_percent])
        top_10_percent_activity = sum(sorted_counts[:top_10_percent])

        print(f"\n=== USER ACTIVITY SKEW ANALYSIS ===")
        print(f"Total activities: {total_activities}")
        print(f"Unique users: {len(user_activity_counts)}")
        print(f"Top 1% users generate: {top_1_percent_activity/total_activities*100:.1f}% of activity")
        print(f"Top 10% users generate: {top_10_percent_activity/total_activities*100:.1f}% of activity")
        print(f"Most active user: {max(sorted_counts)} activities")
        print(f"Least active user: {min(sorted_counts)} activities")
        print(f"Skew ratio (max/avg): {max(sorted_counts)/(total_activities/len(user_activity_counts)):.1f}x")

    def _print_content_skew_analysis(self, engagements: List[Dict], viral_content_ids: List[int]):
        """Print analysis of content engagement skew."""
        content_engagement_counts = {}
        for engagement in engagements:
            content_id = engagement["content_id"]
            content_engagement_counts[content_id] = content_engagement_counts.get(content_id, 0) + 1

        viral_engagement_count = sum(
            content_engagement_counts.get(cid, 0) for cid in viral_content_ids
        )

        print(f"\n=== CONTENT ENGAGEMENT SKEW ANALYSIS ===")
        print(f"Total engagements: {len(engagements)}")
        print(f"Viral content ({len(viral_content_ids)} pieces): {viral_engagement_count/len(engagements)*100:.1f}% of engagements")
        print(f"Viral content ratio: {len(viral_content_ids)/self.total_content*100:.2f}% of all content")

        sorted_engagement_counts = sorted(content_engagement_counts.values(), reverse=True)
        if sorted_engagement_counts:
            print(f"Most engaged content: {max(sorted_engagement_counts)} engagements")
            print(f"Skew ratio: {max(sorted_engagement_counts)/(len(engagements)/len(content_engagement_counts)):.1f}x")


def generate_datasets():
    """Generate all datasets for the social media analytics problem."""
    generator = SocialMediaDataGenerator()

    # Create output directory
    output_dir = "social_media_data"
    os.makedirs(output_dir, exist_ok=True)

    print("🎯 SOCIAL MEDIA ANALYTICS - DATA DISTRIBUTION CHALLENGE")
    print("=" * 60)

    # Generate datasets
    generator.generate_user_activity_logs(10000, output_dir)
    generator.generate_content_engagement(20000, output_dir)

    print(f"\n📁 Generated datasets in '{output_dir}' directory")
    print("\n🎯 CHALLENGES TO IMPLEMENT:")
    print("1. User Influence Score (handle power user skew)")
    print("2. Content Virality Analysis (handle viral content skew)")
    print("3. Hashtag Trending (implement different partitioning strategies)")
    print("4. Load Balancing Visualization")
    print("\n🛠 TECHNIQUES TO EXPLORE:")
    print("- Hash vs Range vs Custom partitioning")
    print("- Hot key detection and salting")
    print("- Combiner optimization")
    print("- Load balancing metrics")


if __name__ == "__main__":
    generate_datasets()
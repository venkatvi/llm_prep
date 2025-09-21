"""
Toy Problem 3: User Activity Aggregation

This demonstrates MapReduce for user behavior analysis:
- Process user activity events (clicks, views, purchases)
- Calculate per-user engagement metrics
- Identify top users and activity patterns

Learning Focus: Multi-dimensional aggregation, complex reduce logic
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path to import scheduler
import sys
sys.path.append(str(Path(__file__).parent.parent))

from mapreduce_scheduler import MapReduceScheduler, JobConfig


def user_activity_map(line: str) -> List[Tuple[str, dict]]:
    """
    Map function for user activity analysis

    Input: JSON line with user activity event
    Output: List of (key, value) pairs for different aggregations
    """
    try:
        event = json.loads(line.strip())
    except json.JSONDecodeError:
        return []

    user_id = event.get('user_id')
    activity_type = event.get('activity_type')
    timestamp = event.get('timestamp')
    value = event.get('value', 0)  # Could be score, price, duration, etc.

    if not user_id or not activity_type:
        return []

    results = []

    # User-level aggregation
    results.append((f"user:{user_id}", {
        'type': 'user_activity',
        'activity_type': activity_type,
        'count': 1,
        'value': value,
        'timestamp': timestamp
    }))

    # Activity type aggregation
    results.append((f"activity:{activity_type}", {
        'type': 'activity_summary',
        'count': 1,
        'value': value,
        'unique_user': user_id
    }))

    # Hourly pattern aggregation
    if timestamp:
        try:
            dt = datetime.fromtimestamp(timestamp)
            hour = dt.hour
            results.append((f"hour:{hour}", {
                'type': 'hourly_pattern',
                'count': 1,
                'value': value
            }))
        except (ValueError, OSError):
            pass

    return results


def user_activity_reduce(key: str, values: List[dict]) -> dict:
    """
    Reduce function for user activity analysis

    Aggregates metrics by key type with complex calculations
    """
    key_type = key.split(':')[0]

    if key_type == 'user':
        # Per-user comprehensive metrics
        activity_counts = {}
        total_value = 0
        total_activities = 0
        timestamps = []

        for value in values:
            activity_type = value['activity_type']
            activity_counts[activity_type] = activity_counts.get(activity_type, 0) + value['count']
            total_value += value['value']
            total_activities += value['count']

            if value['timestamp']:
                timestamps.append(value['timestamp'])

        # Calculate engagement score (weighted by activity type)
        activity_weights = {
            'view': 1,
            'click': 3,
            'share': 5,
            'like': 2,
            'comment': 4,
            'purchase': 10
        }

        engagement_score = sum(
            activity_counts.get(activity, 0) * weight
            for activity, weight in activity_weights.items()
        )

        # Calculate activity span (days between first and last activity)
        activity_span_days = 0
        if len(timestamps) > 1:
            timestamps.sort()
            span_seconds = timestamps[-1] - timestamps[0]
            activity_span_days = span_seconds / (24 * 3600)

        return {
            'total_activities': total_activities,
            'total_value': total_value,
            'avg_value_per_activity': total_value / total_activities if total_activities > 0 else 0,
            'engagement_score': engagement_score,
            'activity_types': activity_counts,
            'activity_span_days': activity_span_days,
            'unique_activity_types': len(activity_counts)
        }

    elif key_type == 'activity':
        # Per-activity type metrics
        total_count = sum(v['count'] for v in values)
        total_value = sum(v['value'] for v in values)
        unique_users = set(v['unique_user'] for v in values)

        return {
            'total_count': total_count,
            'total_value': total_value,
            'avg_value': total_value / total_count if total_count > 0 else 0,
            'unique_users': len(unique_users)
        }

    elif key_type == 'hour':
        # Hourly pattern metrics
        return {
            'total_count': sum(v['count'] for v in values),
            'total_value': sum(v['value'] for v in values),
            'avg_value': sum(v['value'] for v in values) / len(values) if values else 0
        }

    else:
        # Default aggregation
        return {'count': len(values)}


def create_sample_user_activity(output_dir: str, num_files: int = 3, events_per_file: int = 2000) -> List[str]:
    """Create sample user activity data for testing"""

    # Sample data for realistic user behavior
    user_ids = [f"user_{i:04d}" for i in range(1, 101)]  # 100 users

    activity_types = [
        ('view', 0.4),      # 40% of activities
        ('click', 0.25),    # 25% of activities
        ('like', 0.15),     # 15% of activities
        ('share', 0.08),    # 8% of activities
        ('comment', 0.07),  # 7% of activities
        ('purchase', 0.05)  # 5% of activities
    ]

    # Create weighted activity list
    weighted_activities = []
    for activity, weight in activity_types:
        weighted_activities.extend([activity] * int(weight * 100))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    input_files = []
    base_time = datetime.now() - timedelta(days=30)

    for file_idx in range(num_files):
        file_path = output_path / f"user_activity_{file_idx}.json"
        input_files.append(str(file_path))

        with open(file_path, 'w') as f:
            for _ in range(events_per_file):
                # Some users are much more active (Pareto distribution)
                if random.random() < 0.2:  # 20% of users generate 80% of activity
                    user_id = random.choice(user_ids[:20])  # Top 20 users
                else:
                    user_id = random.choice(user_ids)

                activity_type = random.choice(weighted_activities)

                # Generate realistic timestamps (more activity during certain hours)
                hour_weights = [1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8,  # 00-11
                               9, 10, 8, 7, 6, 8, 9, 10, 8, 6, 4, 2]  # 12-23

                hour = random.choices(range(24), weights=hour_weights)[0]
                event_time = base_time + timedelta(
                    days=random.randint(0, 29),
                    hours=hour,
                    minutes=random.randint(0, 59),
                    seconds=random.randint(0, 59)
                )

                # Generate realistic values based on activity type
                if activity_type == 'view':
                    value = random.randint(1, 300)  # View duration in seconds
                elif activity_type == 'click':
                    value = 1  # Binary event
                elif activity_type == 'like':
                    value = 1  # Binary event
                elif activity_type == 'share':
                    value = random.randint(1, 10)  # Reach/influence score
                elif activity_type == 'comment':
                    value = random.randint(1, 5)  # Comment quality score
                elif activity_type == 'purchase':
                    value = random.randint(10, 500)  # Purchase amount
                else:
                    value = 1

                event = {
                    'user_id': user_id,
                    'activity_type': activity_type,
                    'timestamp': int(event_time.timestamp()),
                    'value': value
                }

                f.write(json.dumps(event) + '\n')

    print(f"Created {num_files} sample user activity files with {events_per_file} events each")
    return input_files


def analyze_user_activity_results(output_dir: str):
    """Analyze and pretty-print the user activity results"""

    final_output = Path(output_dir) / "final_output.txt"
    if not final_output.exists():
        print("No output file found!")
        return

    user_metrics = {}
    activity_metrics = {}
    hourly_metrics = {}

    # Parse results
    with open(final_output, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue

            key, value_str = parts
            try:
                value = json.loads(value_str.replace("'", '"'))
            except:
                continue

            key_type = key.split(':')[0]
            key_value = key.split(':', 1)[1]

            if key_type == 'user':
                user_metrics[key_value] = value
            elif key_type == 'activity':
                activity_metrics[key_value] = value
            elif key_type == 'hour':
                hourly_metrics[int(key_value)] = value

    # Display results
    print("\n" + "="*70)
    print("USER ACTIVITY ANALYSIS RESULTS")
    print("="*70)

    # Top users by engagement score
    print("\n🏆 TOP USERS BY ENGAGEMENT SCORE:")
    sorted_users = sorted(user_metrics.items(), key=lambda x: x[1]['engagement_score'], reverse=True)
    for user, metrics in sorted_users[:10]:
        print(f"  {user:10s}: {metrics['engagement_score']:4.0f} engagement, "
              f"{metrics['total_activities']:3d} activities, "
              f"{metrics['unique_activity_types']:1d} types, "
              f"{metrics['activity_span_days']:5.1f} days active")

    # Activity type popularity
    print("\n📊 ACTIVITY TYPE ANALYSIS:")
    sorted_activities = sorted(activity_metrics.items(), key=lambda x: x[1]['total_count'], reverse=True)
    for activity, metrics in sorted_activities:
        print(f"  {activity:10s}: {metrics['total_count']:5d} events, "
              f"{metrics['unique_users']:3d} users, "
              f"avg value: {metrics['avg_value']:6.2f}")

    # Hourly activity patterns
    print("\n⏰ HOURLY ACTIVITY PATTERNS:")
    sorted_hours = sorted(hourly_metrics.items())
    for hour, metrics in sorted_hours:
        print(f"  {hour:2d}:00: {metrics['total_count']:4d} activities, "
              f"avg value: {metrics['avg_value']:6.2f}")

    # User activity distribution analysis
    print("\n📈 USER ACTIVITY DISTRIBUTION:")
    activity_counts = [m['total_activities'] for m in user_metrics.values()]
    engagement_scores = [m['engagement_score'] for m in user_metrics.values()]

    if activity_counts:
        print(f"  Total Users: {len(user_metrics)}")
        print(f"  Avg Activities per User: {sum(activity_counts) / len(activity_counts):.1f}")
        print(f"  Max Activities by Single User: {max(activity_counts)}")
        print(f"  Avg Engagement Score: {sum(engagement_scores) / len(engagement_scores):.1f}")
        print(f"  Max Engagement Score: {max(engagement_scores):.1f}")

        # Power user analysis (top 20% by activity)
        threshold = len(sorted_users) // 5
        power_users = sorted_users[:threshold]
        power_user_activities = sum(m['total_activities'] for _, m in power_users)
        total_activities = sum(activity_counts)

        print(f"  Power Users (top 20%): {len(power_users)} users")
        print(f"  Power User Activity Share: {power_user_activities/total_activities*100:.1f}%")


if __name__ == "__main__":
    print("👥 USER ACTIVITY ANALYSIS MAPREDUCE TOY PROBLEM")
    print("="*55)

    # Create sample user activity data
    data_dir = "/tmp/mapreduce_user_activity"
    input_files = create_sample_user_activity(data_dir, num_files=4, events_per_file=3000)

    # Configure the MapReduce job
    job_config = JobConfig(
        job_name="user_activity_analysis",
        map_function=user_activity_map,
        reduce_function=user_activity_reduce,
        input_files=input_files,
        output_dir="/tmp/mapreduce_user_output",
        num_reduce_tasks=8,  # More partitions for user data
        max_retries=3
    )

    # Run the job
    scheduler = MapReduceScheduler(max_concurrent_tasks=4)
    success = scheduler.execute_job(job_config)

    if success:
        print("\n✅ User activity analysis completed successfully!")
        analyze_user_activity_results(job_config.output_dir)
    else:
        print("\n❌ User activity analysis failed!")
#!/usr/bin/env python3
"""
Level 2 MapReduce Demo - All Toy Problems

This script demonstrates all three toy problems to showcase the Level 2 MapReduce scheduler:
1. Word Count - Basic MapReduce mechanics
2. Log Analysis - Real-world data processing
3. User Activity - Multi-dimensional aggregation

Run this to see the complete Level 2 implementation in action.
"""

import time
import sys
from pathlib import Path

# Import the scheduler and toy problems
from mapreduce_scheduler import (
    MapReduceScheduler, JobConfig,
    word_count_map, word_count_reduce, create_sample_data
)

# Import toy problems
sys.path.append(str(Path(__file__).parent / "toy_problems"))
from log_analysis import (
    log_analysis_map, log_analysis_reduce,
    create_sample_logs, analyze_log_results
)
from user_activity import (
    user_activity_map, user_activity_reduce,
    create_sample_user_activity, analyze_user_activity_results
)


def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")


def run_word_count_demo():
    """Run the word count toy problem"""
    print_section("🔤 PROBLEM 1: WORD COUNT")

    print("Creating sample text data...")
    data_dir = "/tmp/demo_word_count"
    input_files = create_sample_data(data_dir, num_files=3, lines_per_file=800)

    print(f"Input files: {len(input_files)} files")

    # Configure job
    job_config = JobConfig(
        job_name="demo_word_count",
        map_function=word_count_map,
        reduce_function=word_count_reduce,
        input_files=input_files,
        output_dir="/tmp/demo_word_count_output",
        num_reduce_tasks=4
    )

    # Run job
    print("Running MapReduce word count...")
    start_time = time.time()

    scheduler = MapReduceScheduler(max_concurrent_tasks=3)
    success = scheduler.execute_job(job_config)

    duration = time.time() - start_time

    if success:
        print(f"✅ Word count completed in {duration:.2f} seconds")

        # Show sample results
        output_file = Path(job_config.output_dir) / "final_output.txt"
        if output_file.exists():
            print("\n📊 Sample Results (first 10 words):")
            with open(output_file, 'r') as f:
                lines = f.readlines()[:10]
                for line in lines:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        word, count = parts
                        print(f"  {word:15s}: {count:>6s}")
    else:
        print("❌ Word count failed!")

    return success


def run_log_analysis_demo():
    """Run the log analysis toy problem"""
    print_section("🔍 PROBLEM 2: LOG ANALYSIS")

    print("Creating sample web server logs...")
    data_dir = "/tmp/demo_logs"
    input_files = create_sample_logs(data_dir, num_files=3, lines_per_file=1500)

    print(f"Input files: {len(input_files)} log files")

    # Configure job
    job_config = JobConfig(
        job_name="demo_log_analysis",
        map_function=log_analysis_map,
        reduce_function=log_analysis_reduce,
        input_files=input_files,
        output_dir="/tmp/demo_log_output",
        num_reduce_tasks=6
    )

    # Run job
    print("Running MapReduce log analysis...")
    start_time = time.time()

    scheduler = MapReduceScheduler(max_concurrent_tasks=3)
    success = scheduler.execute_job(job_config)

    duration = time.time() - start_time

    if success:
        print(f"✅ Log analysis completed in {duration:.2f} seconds")
        analyze_log_results(job_config.output_dir)
    else:
        print("❌ Log analysis failed!")

    return success


def run_user_activity_demo():
    """Run the user activity toy problem"""
    print_section("👥 PROBLEM 3: USER ACTIVITY ANALYSIS")

    print("Creating sample user activity data...")
    data_dir = "/tmp/demo_user_activity"
    input_files = create_sample_user_activity(data_dir, num_files=3, events_per_file=2000)

    print(f"Input files: {len(input_files)} activity files")

    # Configure job
    job_config = JobConfig(
        job_name="demo_user_activity",
        map_function=user_activity_map,
        reduce_function=user_activity_reduce,
        input_files=input_files,
        output_dir="/tmp/demo_user_output",
        num_reduce_tasks=8
    )

    # Run job
    print("Running MapReduce user activity analysis...")
    start_time = time.time()

    scheduler = MapReduceScheduler(max_concurrent_tasks=4)
    success = scheduler.execute_job(job_config)

    duration = time.time() - start_time

    if success:
        print(f"✅ User activity analysis completed in {duration:.2f} seconds")
        analyze_user_activity_results(job_config.output_dir)
    else:
        print("❌ User activity analysis failed!")

    return success


def print_level2_concepts():
    """Print the Level 2 concepts demonstrated"""
    print_section("🎯 LEVEL 2 CONCEPTS DEMONSTRATED")

    concepts = [
        ("MapReduce Job Scheduler", "✅ Complete map → shuffle → reduce pipeline"),
        ("Intermediate File Management", "✅ Organized file handling between phases"),
        ("Task Coordination", "✅ Dependency management and parallel execution"),
        ("Sequential Processing", "✅ Single-machine implementation"),
        ("File I/O", "✅ Reading/writing large datasets efficiently"),
        ("Memory Management", "✅ File-based data exchange, bounded memory"),
        ("Error Handling", "✅ Task retry logic and failure recovery"),
        ("Concurrent Execution", "✅ Thread-based parallel task processing"),
        ("Data Partitioning", "✅ Hash-based partitioning for reduce tasks"),
        ("Progress Monitoring", "✅ Real-time task status and logging"),
    ]

    for concept, status in concepts:
        print(f"  {status} {concept}")


def print_next_steps():
    """Print suggestions for next steps"""
    print_section("🚀 NEXT STEPS FOR LEVEL 3")

    next_steps = [
        "Memory Management: Implement spillable data structures for large datasets",
        "Streaming I/O: Handle files larger than RAM with streaming processors",
        "Advanced Fault Tolerance: Add checkpointing and recovery mechanisms",
        "Load Balancing: Dynamic task redistribution based on runtime metrics",
        "Combiner Functions: Pre-aggregation to reduce shuffle data volume",
        "Custom Partitioning: Application-specific partitioning strategies",
        "Compression: Compress intermediate files to save disk space",
        "Monitoring: Add detailed performance metrics and profiling"
    ]

    for i, step in enumerate(next_steps, 1):
        print(f"  {i}. {step}")


def main():
    """Run all toy problems and demonstrate Level 2 concepts"""
    print_header("LEVEL 2 MAPREDUCE SCHEDULER DEMONSTRATION")

    print("This demo showcases a complete Level 2 MapReduce implementation with:")
    print("  • Sequential MapReduce job scheduler")
    print("  • Intermediate file management between phases")
    print("  • Task coordination and dependency management")
    print("  • Three toy problems demonstrating different use cases")

    print(f"\nStarting demonstration at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    total_start_time = time.time()
    successes = []

    # Run all three toy problems
    try:
        successes.append(run_word_count_demo())
        successes.append(run_log_analysis_demo())
        successes.append(run_user_activity_demo())

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        return
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {str(e)}")
        return

    total_duration = time.time() - total_start_time

    # Print summary
    print_header("DEMONSTRATION SUMMARY")

    successful_jobs = sum(successes)
    total_jobs = len(successes)

    print(f"📊 Jobs completed: {successful_jobs}/{total_jobs}")
    print(f"⏱️  Total time: {total_duration:.2f} seconds")
    print(f"✅ Success rate: {successful_jobs/total_jobs*100:.1f}%")

    if successful_jobs == total_jobs:
        print("\n🎉 All toy problems completed successfully!")
        print("   The Level 2 MapReduce scheduler is working correctly.")
    else:
        print(f"\n⚠️  {total_jobs - successful_jobs} job(s) failed.")
        print("   Check the logs above for error details.")

    print_level2_concepts()
    print_next_steps()

    print_header("DEMONSTRATION COMPLETE")
    print("Check the output directories for detailed results:")
    print("  • /tmp/demo_word_count_output/")
    print("  • /tmp/demo_log_output/")
    print("  • /tmp/demo_user_output/")


if __name__ == "__main__":
    main()
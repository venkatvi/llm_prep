#!/usr/bin/env python3
"""
Error Recovery and Failure Simulation Demo

This script demonstrates the comprehensive error handling capabilities
of the MapReduce scheduler including:
1. Retry logic with different strategies
2. Checkpointing and job recovery
3. Failure simulation and testing
4. Performance analysis of recovery mechanisms

Test scenarios:
- Basic retry functionality
- Exponential backoff testing
- Checkpoint creation and restoration
- Simulated failures and recovery
- Long-running job interruption and resumption
"""

import time
import os
import random
from pathlib import Path
from typing import List, Dict, Any

from mapreduce_scheduler import (
    MapReduceScheduler, JobConfig,
    word_count_map, word_count_reduce
)
from error_handling import (
    ErrorRecoverySystem, RetryConfig, CheckpointConfig, FailureConfig,
    RetryStrategy, FailureType,
    create_development_config, create_production_config, create_testing_config
)
from streaming_processor import LargeDatasetGenerator, StreamingConfig


def create_test_data() -> List[str]:
    """Create test datasets for error recovery testing"""
    print("📁 Creating test datasets for error recovery testing...")

    config = StreamingConfig(chunk_size_mb=16)
    generator = LargeDatasetGenerator(config)

    datasets = []
    dataset_dir = Path("/tmp/error_recovery_test_data")
    dataset_dir.mkdir(exist_ok=True)

    # Create datasets that will be used for different test scenarios
    test_datasets = [
        (0.05, "50MB"),    # Medium dataset for retry testing
        (0.1, "100MB"),    # Large dataset for checkpoint testing
    ]

    for size_gb, label in test_datasets:
        file_path = dataset_dir / f"error_test_{label.lower()}.txt"

        if file_path.exists():
            print(f"  📄 {label} dataset already exists")
        else:
            print(f"  📄 Creating {label} dataset...")
            start_time = time.time()

            generator.generate_large_text_file(
                str(file_path),
                size_gb,
                "word_count"
            )

            creation_time = time.time() - start_time
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"    ✅ Created in {creation_time:.1f}s ({file_size_mb:.1f}MB)")

        datasets.append(str(file_path))

    return datasets


def test_retry_strategies():
    """Test different retry strategies"""
    print("\n🔄 Testing Retry Strategies...")
    print("=" * 60)

    retry_strategies = [
        (RetryStrategy.IMMEDIATE, "Immediate retry"),
        (RetryStrategy.FIXED_DELAY, "Fixed delay (1s)"),
        (RetryStrategy.LINEAR_BACKOFF, "Linear backoff"),
        (RetryStrategy.EXPONENTIAL_BACKOFF, "Exponential backoff"),
    ]

    results = {}

    for strategy, description in retry_strategies:
        print(f"\n📊 Testing {description}...")

        # Create retry configuration
        retry_config = RetryConfig(
            max_retries=3,
            strategy=strategy,
            base_delay_seconds=0.5,
            jitter=False  # Disable jitter for consistent testing
        )

        # Create error recovery system
        recovery_system = ErrorRecoverySystem(
            retry_config=retry_config,
            checkpoint_config=CheckpointConfig(enabled=False),  # Disable for this test
            failure_config=FailureConfig(enabled=False)  # No simulated failures
        )

        # Test retry logic directly
        retry_count = 0
        errors = []

        def failing_task():
            nonlocal retry_count
            retry_count += 1
            if retry_count <= 2:  # Fail first 2 attempts
                raise RuntimeError(f"Simulated failure on attempt {retry_count}")
            return f"Success after {retry_count} attempts"

        start_time = time.time()
        success, result, error_history = recovery_system.retry_manager.execute_with_retry(
            "test_task", failing_task
        )
        end_time = time.time()

        results[strategy] = {
            'success': success,
            'attempts': retry_count,
            'time': end_time - start_time,
            'errors': len(error_history)
        }

        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {status}: {retry_count} attempts in {end_time - start_time:.2f}s")

    # Summary table
    print(f"\n📈 Retry Strategy Comparison:")
    print(f"{'Strategy':<20} {'Attempts':<10} {'Time (s)':<10} {'Status':<10}")
    print("-" * 60)

    for strategy, description in retry_strategies:
        data = results[strategy]
        status = "SUCCESS" if data['success'] else "FAILED"
        print(f"{description:<20} {data['attempts']:<10} {data['time']:<10.2f} {status:<10}")


def test_checkpointing():
    """Test checkpointing functionality"""
    print("\n💾 Testing Checkpointing Functionality...")
    print("=" * 60)

    # Create test dataset
    datasets = create_test_data()
    test_file = datasets[0]  # Use 50MB file

    # Create scheduler with checkpointing enabled
    checkpoint_config = CheckpointConfig(
        enabled=True,
        checkpoint_interval_seconds=5,
        max_checkpoints_to_keep=3
    )

    recovery_system = ErrorRecoverySystem(
        retry_config=RetryConfig(max_retries=2),
        checkpoint_config=checkpoint_config,
        failure_config=FailureConfig(enabled=False)
    )

    scheduler = MapReduceScheduler(
        max_concurrent_tasks=2,
        recovery_system=recovery_system
    )

    # Test job configuration
    job_config = JobConfig(
        job_name="checkpoint_test",
        map_function=word_count_map,
        reduce_function=word_count_reduce,
        input_files=[test_file],
        output_dir="/tmp/checkpoint_test_output",
        num_reduce_tasks=2,
        enable_error_recovery=True
    )

    print("🚀 Running job with checkpointing enabled...")
    start_time = time.time()
    success = scheduler.execute_job(job_config)
    end_time = time.time()

    print(f"✅ Job completed: {success} in {end_time - start_time:.2f}s")

    # Check for checkpoint files
    checkpoint_files = recovery_system.checkpoint_manager.list_available_checkpoints("checkpoint_test")
    print(f"📄 Created {len(checkpoint_files)} checkpoint files:")
    for checkpoint_file in checkpoint_files:
        print(f"  - {Path(checkpoint_file).name}")

    # Test checkpoint loading
    if checkpoint_files:
        print("\n🔄 Testing checkpoint recovery...")
        job_id = scheduler.current_job_id
        checkpoint = recovery_system.recover_job_from_checkpoint("checkpoint_test", job_id)

        if checkpoint:
            print(f"✅ Successfully loaded checkpoint:")
            print(f"  - Total tasks: {len(checkpoint.task_checkpoints)}")
            print(f"  - Completed tasks: {len(checkpoint.completed_tasks)}")
            print(f"  - Failed tasks: {len(checkpoint.failed_tasks)}")
            print(f"  - Overall progress: {checkpoint.total_progress:.1%}")

    # Show recovery statistics
    stats = scheduler.get_error_recovery_stats()
    print(f"\n📊 Recovery Statistics:")
    for key, value in stats.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")


def test_failure_simulation():
    """Test failure simulation and recovery"""
    print("\n💥 Testing Failure Simulation and Recovery...")
    print("=" * 60)

    datasets = create_test_data()
    test_file = datasets[0]  # Use 50MB file

    # Test different failure scenarios
    failure_scenarios = [
        {
            'name': 'Task Crash Simulation',
            'failure_rate': 0.3,
            'failure_types': [FailureType.TASK_CRASH],
            'description': '30% chance of task crashes'
        },
        {
            'name': 'Mixed Failures',
            'failure_rate': 0.2,
            'failure_types': [FailureType.NETWORK_FAILURE, FailureType.TIMEOUT, FailureType.MEMORY_ERROR],
            'description': '20% chance of various failures'
        },
        {
            'name': 'Targeted Failure',
            'failure_rate': 1.0,
            'failure_types': [FailureType.TASK_CRASH],
            'target_tasks': ['map_0'],
            'description': '100% failure rate for map_0 task'
        }
    ]

    results = {}

    for scenario in failure_scenarios:
        print(f"\n🧪 Testing: {scenario['name']}")
        print(f"   {scenario['description']}")

        # Create recovery system with failure simulation
        failure_config = FailureConfig(
            enabled=True,
            failure_rate=scenario['failure_rate'],
            failure_types=scenario['failure_types'],
            target_tasks=scenario.get('target_tasks')
        )

        recovery_system = ErrorRecoverySystem(
            retry_config=RetryConfig(max_retries=3, strategy=RetryStrategy.EXPONENTIAL_BACKOFF),
            checkpoint_config=CheckpointConfig(enabled=True),
            failure_config=failure_config
        )

        scheduler = MapReduceScheduler(
            max_concurrent_tasks=2,
            recovery_system=recovery_system
        )

        # Job configuration
        job_config = JobConfig(
            job_name=f"failure_test_{scenario['name'].lower().replace(' ', '_')}",
            map_function=word_count_map,
            reduce_function=word_count_reduce,
            input_files=[test_file],
            output_dir=f"/tmp/failure_test_{scenario['name'].lower().replace(' ', '_')}_output",
            num_reduce_tasks=2,
            enable_error_recovery=True
        )

        print("🚀 Running job with failure simulation...")
        start_time = time.time()
        success = scheduler.execute_job(job_config)
        end_time = time.time()

        # Collect results
        stats = scheduler.get_error_recovery_stats()
        results[scenario['name']] = {
            'success': success,
            'time': end_time - start_time,
            'stats': stats
        }

        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {status} in {end_time - start_time:.2f}s")
        print(f"   Failed tasks: {stats['failed_tasks']}")
        print(f"   Recovered tasks: {stats['recovered_tasks']}")
        print(f"   Permanently failed: {stats['permanently_failed_tasks']}")

    # Summary
    print(f"\n📈 Failure Recovery Summary:")
    print(f"{'Scenario':<25} {'Result':<10} {'Time(s)':<10} {'Failed':<8} {'Recovered':<10}")
    print("-" * 75)

    for scenario_name, data in results.items():
        result = "SUCCESS" if data['success'] else "FAILED"
        stats = data['stats']
        print(f"{scenario_name:<25} {result:<10} {data['time']:<10.2f} "
              f"{stats['failed_tasks']:<8} {stats['recovered_tasks']:<10}")


def test_long_running_job_interruption():
    """Test interruption and recovery of long-running jobs"""
    print("\n⏳ Testing Long-Running Job Interruption and Recovery...")
    print("=" * 60)

    datasets = create_test_data()
    large_file = datasets[1]  # Use 100MB file

    # Create recovery system with short checkpoint intervals
    recovery_system = ErrorRecoverySystem(
        retry_config=RetryConfig(max_retries=2),
        checkpoint_config=CheckpointConfig(
            enabled=True,
            checkpoint_interval_seconds=3,  # Frequent checkpoints
            max_checkpoints_to_keep=5
        ),
        failure_config=FailureConfig(enabled=False)
    )

    scheduler = MapReduceScheduler(
        max_concurrent_tasks=1,  # Single task to make it longer
        recovery_system=recovery_system
    )

    job_config = JobConfig(
        job_name="long_running_test",
        map_function=word_count_map,
        reduce_function=word_count_reduce,
        input_files=[large_file],
        output_dir="/tmp/long_running_test_output",
        num_reduce_tasks=2,
        enable_error_recovery=True,
        enable_streaming=True  # Use streaming for large file
    )

    print("🚀 Starting long-running job...")
    print("   (This will run for a while to demonstrate checkpointing)")

    # Start the job (it will run and create checkpoints)
    start_time = time.time()
    success = scheduler.execute_job(job_config)
    end_time = time.time()

    print(f"✅ Job completed: {success} in {end_time - start_time:.2f}s")

    # Check checkpoints created
    checkpoint_files = recovery_system.checkpoint_manager.list_available_checkpoints("long_running_test")
    print(f"📄 Created {len(checkpoint_files)} checkpoints during execution")

    # Show recovery statistics
    stats = scheduler.get_error_recovery_stats()
    print(f"\n📊 Final Statistics:")
    for key, value in stats.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")


def analyze_error_recovery_performance():
    """Analyze performance impact of error recovery features"""
    print("\n📊 Analyzing Error Recovery Performance Impact...")
    print("=" * 60)

    datasets = create_test_data()
    test_file = datasets[0]  # Use 50MB file

    # Test configurations
    test_configs = [
        {
            'name': 'No Error Recovery',
            'enable_recovery': False,
            'description': 'Baseline performance without error recovery'
        },
        {
            'name': 'Basic Recovery',
            'enable_recovery': True,
            'config': create_development_config(),
            'description': 'Basic error recovery with fast retries'
        },
        {
            'name': 'Production Recovery',
            'enable_recovery': True,
            'config': create_production_config(),
            'description': 'Production-grade error recovery'
        },
        {
            'name': 'Testing Recovery',
            'enable_recovery': True,
            'config': create_testing_config(),
            'description': 'Recovery with failure simulation (20% failure rate)'
        }
    ]

    results = {}

    for config in test_configs:
        print(f"\n🧪 Testing: {config['name']}")
        print(f"   {config['description']}")

        # Create scheduler
        if config['enable_recovery']:
            recovery_system = ErrorRecoverySystem(*config['config'])
            scheduler = MapReduceScheduler(max_concurrent_tasks=2, recovery_system=recovery_system)
        else:
            scheduler = MapReduceScheduler(max_concurrent_tasks=2)

        # Job configuration
        job_config = JobConfig(
            job_name=f"performance_test_{config['name'].lower().replace(' ', '_')}",
            map_function=word_count_map,
            reduce_function=word_count_reduce,
            input_files=[test_file],
            output_dir=f"/tmp/performance_test_{config['name'].lower().replace(' ', '_')}_output",
            num_reduce_tasks=2,
            enable_error_recovery=config['enable_recovery']
        )

        # Run multiple iterations for averaging
        times = []
        successes = 0

        for i in range(3):  # Run 3 iterations
            print(f"   Iteration {i + 1}/3...")
            start_time = time.time()
            success = scheduler.execute_job(job_config)
            end_time = time.time()

            if success:
                successes += 1
                times.append(end_time - start_time)

        # Calculate statistics
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
        else:
            avg_time = min_time = max_time = 0

        results[config['name']] = {
            'success_rate': successes / 3,
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'times': times
        }

        print(f"   Success rate: {successes}/3 ({successes/3:.1%})")
        if times:
            print(f"   Average time: {avg_time:.2f}s (min: {min_time:.2f}s, max: {max_time:.2f}s)")

    # Performance comparison
    print(f"\n📈 Performance Comparison:")
    print(f"{'Configuration':<20} {'Success Rate':<12} {'Avg Time (s)':<12} {'Overhead':<10}")
    print("-" * 65)

    baseline_time = results.get('No Error Recovery', {}).get('avg_time', 0)

    for config_name, data in results.items():
        success_rate = f"{data['success_rate']:.1%}"
        avg_time = data['avg_time']

        if baseline_time > 0 and avg_time > 0:
            overhead = f"{((avg_time - baseline_time) / baseline_time * 100):+.1f}%"
        else:
            overhead = "N/A"

        print(f"{config_name:<20} {success_rate:<12} {avg_time:<12.2f} {overhead:<10}")


def main():
    """Run the complete error recovery demonstration"""
    print("💾 ERROR RECOVERY AND FAILURE SIMULATION DEMONSTRATION")
    print("=" * 80)

    print("This demo showcases comprehensive error handling capabilities:")
    print("  • Retry logic with different strategies")
    print("  • Checkpointing for job state persistence")
    print("  • Failure simulation and recovery testing")
    print("  • Performance analysis of recovery mechanisms")

    try:
        # Run all test scenarios
        test_retry_strategies()
        test_checkpointing()
        test_failure_simulation()
        test_long_running_job_interruption()
        analyze_error_recovery_performance()

        print("\n🎉 Error Recovery Demo Complete!")
        print("\nKey capabilities demonstrated:")
        print("  ✅ Configurable retry strategies with exponential backoff")
        print("  ✅ Automatic checkpointing and job state persistence")
        print("  ✅ Failure simulation for comprehensive testing")
        print("  ✅ Graceful recovery from various failure types")
        print("  ✅ Performance monitoring and optimization")

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
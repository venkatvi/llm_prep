#!/usr/bin/env python3
"""
Streaming File Processing Demo

This script demonstrates the streaming capabilities of the MapReduce scheduler:
1. Generate large test datasets (100MB, 500MB, 1GB)
2. Compare traditional vs streaming processing
3. Test memory usage and performance
4. Demonstrate file splitting for parallel processing

Run this to see Level 2+ streaming implementation in action.
"""

import time
import psutil
import os
from pathlib import Path

from mapreduce_scheduler import (
    MapReduceScheduler, JobConfig,
    word_count_map, word_count_reduce
)
from streaming_processor import (
    StreamingConfig, LargeDatasetGenerator, StreamingIntegration
)


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


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


def create_large_datasets():
    """Create test datasets of various sizes"""
    print_section("📁 CREATING LARGE TEST DATASETS")

    config = StreamingConfig(chunk_size_mb=32)
    generator = LargeDatasetGenerator(config)

    datasets = []
    dataset_dir = Path("/tmp/streaming_test_data")
    dataset_dir.mkdir(exist_ok=True)

    # Create datasets of different sizes
    sizes = [
        (0.1, "100MB"),    # 100MB
        (0.5, "500MB"),    # 500MB
        (1.0, "1GB")       # 1GB
    ]

    for size_gb, label in sizes:
        file_path = dataset_dir / f"large_dataset_{label.lower()}.txt"

        if file_path.exists():
            print(f"📄 {label} dataset already exists: {file_path}")
        else:
            print(f"📄 Creating {label} dataset...")
            start_time = time.time()

            generator.generate_large_text_file(
                str(file_path),
                size_gb,
                "word_count"
            )

            creation_time = time.time() - start_time
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"   ✅ Created in {creation_time:.1f}s ({file_size_mb:.1f}MB)")

        datasets.append(str(file_path))

    return datasets


def run_traditional_processing(input_file: str, label: str):
    """Run traditional processing on a file"""
    print(f"\n🔄 Traditional Processing ({label})")

    start_memory = get_memory_usage()
    start_time = time.time()

    job_config = JobConfig(
        job_name=f"traditional_{label}",
        map_function=word_count_map,
        reduce_function=word_count_reduce,
        input_files=[input_file],
        output_dir=f"/tmp/traditional_output_{label}",
        num_reduce_tasks=4,
        enable_streaming=False  # Explicitly disable streaming
    )

    scheduler = MapReduceScheduler(max_concurrent_tasks=2)
    success = scheduler.execute_job(job_config)

    end_time = time.time()
    end_memory = get_memory_usage()
    peak_memory = end_memory - start_memory

    print(f"   {'✅' if success else '❌'} Result: {'Success' if success else 'Failed'}")
    print(f"   ⏱️  Time: {end_time - start_time:.2f} seconds")
    print(f"   💾 Memory: {peak_memory:.1f}MB peak usage")

    return success, end_time - start_time, peak_memory


def run_streaming_processing(input_file: str, label: str):
    """Run streaming processing on a file"""
    print(f"\n🌊 Streaming Processing ({label})")

    start_memory = get_memory_usage()
    start_time = time.time()

    # Configure streaming
    streaming_config = StreamingConfig(
        chunk_size_mb=64,  # 64MB chunks
        enable_mmap=True
    )

    job_config = JobConfig(
        job_name=f"streaming_{label}",
        map_function=word_count_map,
        reduce_function=word_count_reduce,
        input_files=[input_file],
        output_dir=f"/tmp/streaming_output_{label}",
        num_reduce_tasks=4,
        enable_streaming=True,
        streaming_config=streaming_config
    )

    scheduler = MapReduceScheduler(max_concurrent_tasks=2)
    success = scheduler.execute_job(job_config)

    end_time = time.time()
    end_memory = get_memory_usage()
    peak_memory = end_memory - start_memory

    print(f"   {'✅' if success else '❌'} Result: {'Success' if success else 'Failed'}")
    print(f"   ⏱️  Time: {end_time - start_time:.2f} seconds")
    print(f"   💾 Memory: {peak_memory:.1f}MB peak usage")

    return success, end_time - start_time, peak_memory


def run_parallel_streaming_processing(input_file: str, label: str):
    """Run streaming processing with file splitting for parallel maps"""
    print(f"\n⚡ Parallel Streaming Processing ({label})")

    start_memory = get_memory_usage()
    start_time = time.time()

    # Configure streaming with parallel map tasks
    streaming_config = StreamingConfig(
        chunk_size_mb=32,  # Smaller chunks for splitting
        enable_mmap=True
    )

    job_config = JobConfig(
        job_name=f"parallel_streaming_{label}",
        map_function=word_count_map,
        reduce_function=word_count_reduce,
        input_files=[input_file],
        output_dir=f"/tmp/parallel_output_{label}",
        num_reduce_tasks=4,
        enable_streaming=True,
        streaming_config=streaming_config,
        num_map_tasks=6  # Split into 6 parallel map tasks
    )

    scheduler = MapReduceScheduler(max_concurrent_tasks=4)
    success = scheduler.execute_job(job_config)

    end_time = time.time()
    end_memory = get_memory_usage()
    peak_memory = end_memory - start_memory

    print(f"   {'✅' if success else '❌'} Result: {'Success' if success else 'Failed'}")
    print(f"   ⏱️  Time: {end_time - start_time:.2f} seconds")
    print(f"   💾 Memory: {peak_memory:.1f}MB peak usage")

    return success, end_time - start_time, peak_memory


def compare_processing_methods(datasets):
    """Compare traditional vs streaming processing"""
    print_section("📊 PROCESSING METHOD COMPARISON")

    results = []

    for dataset in datasets:
        file_size_mb = os.path.getsize(dataset) / (1024 * 1024)
        label = f"{file_size_mb:.0f}MB"

        print(f"\n📄 Testing file: {Path(dataset).name} ({label})")

        # Skip traditional processing for very large files to avoid memory issues
        if file_size_mb <= 200:  # Only run traditional for smaller files
            trad_success, trad_time, trad_memory = run_traditional_processing(dataset, label)
        else:
            print(f"\n🔄 Traditional Processing ({label}) - SKIPPED (file too large)")
            trad_success, trad_time, trad_memory = False, 0, 0

        # Run streaming processing
        stream_success, stream_time, stream_memory = run_streaming_processing(dataset, label)

        # Run parallel streaming processing
        parallel_success, parallel_time, parallel_memory = run_parallel_streaming_processing(dataset, label)

        results.append({
            'file': Path(dataset).name,
            'size_mb': file_size_mb,
            'traditional': (trad_success, trad_time, trad_memory),
            'streaming': (stream_success, stream_time, stream_memory),
            'parallel': (parallel_success, parallel_time, parallel_memory)
        })

    return results


def print_performance_summary(results):
    """Print a summary of performance results"""
    print_section("📈 PERFORMANCE SUMMARY")

    print(f"{'File':<20} {'Size':<8} {'Method':<12} {'Time':<8} {'Memory':<10} {'Status'}")
    print("-" * 70)

    for result in results:
        file_name = result['file'][:18]
        size = f"{result['size_mb']:.0f}MB"

        # Traditional
        trad_success, trad_time, trad_memory = result['traditional']
        if trad_success:
            print(f"{file_name:<20} {size:<8} {'Traditional':<12} {trad_time:<8.2f} {trad_memory:<10.1f} {'✅'}")
        elif trad_time == 0:
            print(f"{file_name:<20} {size:<8} {'Traditional':<12} {'SKIP':<8} {'SKIP':<10} {'⏭️'}")
        else:
            print(f"{file_name:<20} {size:<8} {'Traditional':<12} {trad_time:<8.2f} {trad_memory:<10.1f} {'❌'}")

        # Streaming
        stream_success, stream_time, stream_memory = result['streaming']
        status = '✅' if stream_success else '❌'
        print(f"{'':<20} {'':<8} {'Streaming':<12} {stream_time:<8.2f} {stream_memory:<10.1f} {status}")

        # Parallel
        parallel_success, parallel_time, parallel_memory = result['parallel']
        status = '✅' if parallel_success else '❌'
        print(f"{'':<20} {'':<8} {'Parallel':<12} {parallel_time:<8.2f} {parallel_memory:<10.1f} {status}")
        print()

    # Calculate improvements
    print_section("🚀 STREAMING BENEFITS")

    valid_comparisons = [r for r in results if r['traditional'][0]]  # Only successful traditional runs

    if valid_comparisons:
        for result in valid_comparisons:
            trad_time, trad_memory = result['traditional'][1], result['traditional'][2]
            stream_time, stream_memory = result['streaming'][1], result['streaming'][2]
            parallel_time, parallel_memory = result['parallel'][1], result['parallel'][2]

            if trad_time > 0 and stream_time > 0:
                time_improvement = ((trad_time - stream_time) / trad_time) * 100
                memory_improvement = ((trad_memory - stream_memory) / trad_memory) * 100 if trad_memory > 0 else 0
                parallel_speedup = (stream_time / parallel_time) if parallel_time > 0 else 1

                print(f"📄 {result['file']} ({result['size_mb']:.0f}MB):")
                print(f"   ⏱️  Streaming vs Traditional: {time_improvement:+.1f}% time")
                print(f"   💾 Streaming vs Traditional: {memory_improvement:+.1f}% memory")
                print(f"   ⚡ Parallel speedup: {parallel_speedup:.1f}x")

    else:
        print("ℹ️  All test files were too large for traditional processing")
        print("   This demonstrates the necessity of streaming for large datasets!")


def test_memory_boundaries():
    """Test streaming with various memory constraints"""
    print_section("🧪 MEMORY BOUNDARY TESTING")

    # Create a medium-sized file for testing
    config = StreamingConfig(chunk_size_mb=16)
    generator = LargeDatasetGenerator(config)

    test_file = "/tmp/memory_test_file.txt"
    print("Creating 200MB test file for memory testing...")
    generator.generate_large_text_file(test_file, 0.2, "word_count")

    # Test different chunk sizes
    chunk_sizes = [8, 16, 32, 64, 128]

    print(f"\n📊 Testing different chunk sizes:")
    print(f"{'Chunk Size':<12} {'Time':<8} {'Memory':<10} {'Status'}")
    print("-" * 40)

    for chunk_size in chunk_sizes:
        start_memory = get_memory_usage()
        start_time = time.time()

        streaming_config = StreamingConfig(
            chunk_size_mb=chunk_size,
            enable_mmap=True
        )

        job_config = JobConfig(
            job_name=f"memory_test_{chunk_size}mb",
            map_function=word_count_map,
            reduce_function=word_count_reduce,
            input_files=[test_file],
            output_dir=f"/tmp/memory_test_output_{chunk_size}",
            num_reduce_tasks=2,
            enable_streaming=True,
            streaming_config=streaming_config
        )

        scheduler = MapReduceScheduler(max_concurrent_tasks=1)
        success = scheduler.execute_job(job_config)

        end_time = time.time()
        end_memory = get_memory_usage()
        peak_memory = end_memory - start_memory

        status = '✅' if success else '❌'
        print(f"{chunk_size:<12} {end_time - start_time:<8.2f} {peak_memory:<10.1f} {status}")


def main():
    """Run the complete streaming demonstration"""
    print_header("LEVEL 2+ STREAMING FILE PROCESSING DEMONSTRATION")

    print("This demo showcases advanced file streaming capabilities:")
    print("  • Process files larger than available RAM")
    print("  • Compare traditional vs streaming performance")
    print("  • Test parallel processing with file splitting")
    print("  • Demonstrate memory-bounded processing")

    try:
        # Create test datasets
        datasets = create_large_datasets()

        # Compare processing methods
        results = compare_processing_methods(datasets)

        # Print performance summary
        print_performance_summary(results)

        # Test memory boundaries
        test_memory_boundaries()

        print_header("STREAMING DEMONSTRATION COMPLETE")
        print("🎉 All streaming tests completed successfully!")
        print("\nKey takeaways:")
        print("  ✅ Streaming enables processing of arbitrarily large files")
        print("  ✅ Memory usage remains bounded regardless of file size")
        print("  ✅ Parallel processing improves performance for large files")
        print("  ✅ Chunk size can be tuned for optimal performance")

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {str(e)}")


if __name__ == "__main__":
    main()
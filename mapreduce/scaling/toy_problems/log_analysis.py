"""
Toy Problem 2: Web Server Log Analysis

This demonstrates MapReduce for real-world log processing:
- Parse web server logs
- Extract IP addresses, response codes, and timestamps
- Count requests per IP and calculate error rates

Learning Focus: Complex parsing, multiple output types
"""

import json
import random
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path to import scheduler
import sys
sys.path.append(str(Path(__file__).parent.parent))

from mapreduce_scheduler import MapReduceScheduler, JobConfig


def log_analysis_map(line: str) -> List[Tuple[str, dict]]:
    """
    Map function for log analysis

    Input: Log line in Common Log Format
    Output: List of (key, value) pairs for different metrics
    """
    # Common Log Format regex
    # IP - - [timestamp] "method path protocol" status size
    log_pattern = r'^(\S+) \S+ \S+ \[([^\]]+)\] "(\S+) (\S+) (\S+)" (\d+) (\d+)'

    match = re.match(log_pattern, line.strip())
    if not match:
        return []

    ip, timestamp, method, path, protocol, status, size = match.groups()
    status_code = int(status)
    response_size = int(size) if size != '-' else 0

    results = []

    # Emit IP-based metrics
    results.append((f"ip:{ip}", {
        'type': 'request_count',
        'count': 1,
        'bytes': response_size,
        'is_error': 1 if status_code >= 400 else 0
    }))

    # Emit status code metrics
    results.append((f"status:{status_code}", {
        'type': 'status_count',
        'count': 1
    }))

    # Emit hourly metrics
    try:
        # Parse timestamp: [01/Jan/2024:12:00:00 +0000]
        dt = datetime.strptime(timestamp.split()[0], "%d/%b/%Y:%H:%M:%S")
        hour_key = dt.strftime("%Y-%m-%d_%H")

        results.append((f"hour:{hour_key}", {
            'type': 'hourly_count',
            'count': 1,
            'bytes': response_size
        }))
    except ValueError:
        pass  # Skip invalid timestamps

    return results


def log_analysis_reduce(key: str, values: List[dict]) -> dict:
    """
    Reduce function for log analysis

    Aggregates metrics by key type
    """
    key_type = key.split(':')[0]

    if key_type == 'ip':
        # Aggregate per-IP metrics
        total_requests = sum(v['count'] for v in values)
        total_bytes = sum(v['bytes'] for v in values)
        total_errors = sum(v['is_error'] for v in values)

        return {
            'requests': total_requests,
            'bytes': total_bytes,
            'errors': total_errors,
            'error_rate': total_errors / total_requests if total_requests > 0 else 0
        }

    elif key_type == 'status':
        # Aggregate status code counts
        return {
            'count': sum(v['count'] for v in values)
        }

    elif key_type == 'hour':
        # Aggregate hourly metrics
        return {
            'requests': sum(v['count'] for v in values),
            'bytes': sum(v['bytes'] for v in values)
        }

    else:
        # Default aggregation
        return {'count': len(values)}


def create_sample_logs(output_dir: str, num_files: int = 3, lines_per_file: int = 1000) -> List[str]:
    """Create sample web server logs for testing"""

    # Sample data for realistic logs
    ips = [
        '192.168.1.100', '192.168.1.101', '192.168.1.102',
        '10.0.0.50', '10.0.0.51', '203.0.113.1', '203.0.113.2',
        '198.51.100.1', '198.51.100.2', '172.16.0.1'
    ]

    methods = ['GET', 'POST', 'PUT', 'DELETE']
    paths = [
        '/index.html', '/about.html', '/contact.html', '/api/users',
        '/api/products', '/images/logo.png', '/css/style.css',
        '/js/script.js', '/api/orders', '/admin/dashboard'
    ]

    status_codes = [200, 200, 200, 200, 304, 404, 500, 403, 301]  # Weighted toward 200

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    input_files = []
    base_time = datetime.now() - timedelta(days=1)

    for file_idx in range(num_files):
        file_path = output_path / f"access_log_{file_idx}.log"
        input_files.append(str(file_path))

        with open(file_path, 'w') as f:
            for line_idx in range(lines_per_file):
                # Generate realistic log entry
                ip = random.choice(ips)

                # Some IPs are more active (simulate real traffic patterns)
                if ip in ['192.168.1.100', '10.0.0.50']:
                    weight = 3  # 3x more likely to appear
                else:
                    weight = 1

                # Skip some entries based on weight
                if random.randint(1, weight) != 1:
                    continue

                timestamp = base_time + timedelta(
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59),
                    seconds=random.randint(0, 59)
                )
                timestamp_str = timestamp.strftime("%d/%b/%Y:%H:%M:%S +0000")

                method = random.choice(methods)
                path = random.choice(paths)
                status = random.choice(status_codes)
                size = random.randint(100, 50000) if status == 200 else random.randint(0, 1000)

                # Format: IP - - [timestamp] "method path protocol" status size
                log_line = f'{ip} - - [{timestamp_str}] "{method} {path} HTTP/1.1" {status} {size}'
                f.write(log_line + '\n')

    print(f"Created {num_files} sample log files with approximately {lines_per_file} entries each")
    return input_files


def analyze_log_results(output_dir: str):
    """Analyze and pretty-print the log analysis results"""

    final_output = Path(output_dir) / "final_output.txt"
    if not final_output.exists():
        print("No output file found!")
        return

    ip_metrics = {}
    status_metrics = {}
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

            if key_type == 'ip':
                ip_metrics[key_value] = value
            elif key_type == 'status':
                status_metrics[key_value] = value
            elif key_type == 'hour':
                hourly_metrics[key_value] = value

    # Display results
    print("\n" + "="*60)
    print("WEB SERVER LOG ANALYSIS RESULTS")
    print("="*60)

    # Top IPs by request count
    print("\n📊 TOP IP ADDRESSES BY REQUEST COUNT:")
    sorted_ips = sorted(ip_metrics.items(), key=lambda x: x[1]['requests'], reverse=True)
    for ip, metrics in sorted_ips[:10]:
        print(f"  {ip:15s}: {metrics['requests']:5d} requests, "
              f"{metrics['bytes']:8d} bytes, "
              f"{metrics['error_rate']*100:5.1f}% errors")

    # Status code distribution
    print("\n📈 HTTP STATUS CODE DISTRIBUTION:")
    sorted_status = sorted(status_metrics.items(), key=lambda x: int(x[0]))
    for status, metrics in sorted_status:
        print(f"  HTTP {status}: {metrics['count']:4d} responses")

    # Hourly traffic patterns
    print("\n⏰ HOURLY TRAFFIC PATTERNS:")
    sorted_hours = sorted(hourly_metrics.items())
    for hour, metrics in sorted_hours[:10]:  # Show first 10 hours
        print(f"  {hour}: {metrics['requests']:4d} requests, "
              f"{metrics['bytes']:8d} bytes")

    # Summary statistics
    total_requests = sum(m['requests'] for m in ip_metrics.values())
    total_bytes = sum(m['bytes'] for m in ip_metrics.values())
    total_errors = sum(m['errors'] for m in ip_metrics.values())

    print(f"\n📋 SUMMARY STATISTICS:")
    print(f"  Total Requests: {total_requests:,}")
    print(f"  Total Bytes: {total_bytes:,}")
    print(f"  Total Errors: {total_errors:,}")
    print(f"  Overall Error Rate: {total_errors/total_requests*100:.2f}%")
    print(f"  Unique IPs: {len(ip_metrics)}")


if __name__ == "__main__":
    print("🔍 LOG ANALYSIS MAPREDUCE TOY PROBLEM")
    print("="*50)

    # Create sample log data
    data_dir = "/tmp/mapreduce_logs"
    input_files = create_sample_logs(data_dir, num_files=4, lines_per_file=2000)

    # Configure the MapReduce job
    job_config = JobConfig(
        job_name="log_analysis",
        map_function=log_analysis_map,
        reduce_function=log_analysis_reduce,
        input_files=input_files,
        output_dir="/tmp/mapreduce_log_output",
        num_reduce_tasks=6,  # More partitions for better parallelism
        max_retries=3
    )

    # Run the job
    scheduler = MapReduceScheduler(max_concurrent_tasks=3)
    success = scheduler.execute_job(job_config)

    if success:
        print("\n✅ Log analysis completed successfully!")
        analyze_log_results(job_config.output_dir)
    else:
        print("\n❌ Log analysis failed!")
# Level 2 MapReduce Scheduler

This directory contains a complete implementation of a sequential MapReduce job scheduler that demonstrates Level 2 concepts:

## 🎯 Core Features Implemented

### 1. MapReduce Job Scheduler (`MapReduceScheduler`)
- **Map → Shuffle → Reduce Phases**: Complete execution pipeline
- **Task Coordination**: Dependency management between phases
- **Concurrent Execution**: Thread-based parallel task execution
- **Progress Monitoring**: Real-time task status tracking

### 2. Intermediate File Management (`IntermediateFileManager`)
- **Phase Separation**: Organized directories for map/shuffle/reduce outputs
- **File Coordination**: Automatic file path management between phases
- **Cleanup**: Automatic cleanup of intermediate files
- **Partition Management**: Handles data partitioning for reduce tasks

### 3. Task Coordination (`TaskCoordinator`)
- **Dependency Tracking**: Tasks wait for prerequisites to complete
- **Status Management**: Tracks pending/running/completed/failed states
- **Retry Logic**: Built-in retry mechanisms for failed tasks
- **Progress Reporting**: Statistics on task execution

## 📁 File Structure

```
scaling/
├── mapreduce_scheduler.py    # Main scheduler implementation
├── README.md                 # This file
├── toy_problems/            # Additional examples
│   ├── word_count_demo.py   # Simple word count example
│   ├── log_analysis.py      # Log file analysis
│   └── user_activity.py     # User activity aggregation
└── test_data/               # Sample datasets
    ├── small/               # Small test files
    ├── medium/              # Medium-sized datasets
    └── large/               # Large datasets for stress testing
```

## 🚀 Quick Start

### Basic Word Count Example

```python
from mapreduce_scheduler import MapReduceScheduler, JobConfig, word_count_map, word_count_reduce

# Configure job
job_config = JobConfig(
    job_name="word_count",
    map_function=word_count_map,
    reduce_function=word_count_reduce,
    input_files=["input1.txt", "input2.txt"],
    output_dir="./output",
    num_reduce_tasks=4
)

# Run job
scheduler = MapReduceScheduler(max_concurrent_tasks=2)
success = scheduler.execute_job(job_config)
```

### Running the Built-in Example

```bash
cd /Users/arvindsudarsanam/code/prep/mapreduce/scaling
python mapreduce_scheduler.py
```

This will:
1. Create sample text data (3 files, 500 lines each)
2. Run word count MapReduce job
3. Output results to `/tmp/mapreduce_output/final_output.txt`

## 🔧 Key Components Explained

### Task Dependency Management

The scheduler handles complex dependencies:

```
Map Tasks (parallel)     →  Shuffle Task  →  Reduce Tasks (parallel)
├── map_0               ↗                 ↗  ├── reduce_0
├── map_1               →     shuffle     →   ├── reduce_1
├── map_2               ↘                 ↘  ├── reduce_2
└── map_3                                    └── reduce_3
```

### Intermediate File Organization

```
intermediate/
├── map_output/          # Raw map task outputs
│   ├── map_0.json
│   ├── map_1.json
│   └── ...
├── shuffle/             # Temporary shuffle files
│   ├── shuffle_partition_0.json
│   └── ...
└── reduce_input/        # Sorted, partitioned data for reduce
    ├── reduce_input_reduce_0.json
    └── ...
```

### Phase Execution Details

1. **Map Phase**:
   - Each input file gets one map task
   - Map function applied to each line
   - Output: JSON records with partition info

2. **Shuffle Phase**:
   - Collects all map outputs
   - Partitions data by key hash
   - Sorts within each partition
   - Creates reduce input files

3. **Reduce Phase**:
   - Groups values by key
   - Applies reduce function
   - Outputs final results

## 🎮 Toy Problems

### 1. Word Count (`mapreduce_scheduler.py`)
**Learning Focus**: Basic MapReduce mechanics
- **Input**: Text files
- **Map**: `(line) → [(word, 1), ...]`
- **Reduce**: `(word, [1,1,1,...]) → (word, count)`

### 2. Log Analysis (`toy_problems/log_analysis.py`)
**Learning Focus**: Real-world data processing
- **Input**: Web server logs
- **Map**: Extract IP addresses and response codes
- **Reduce**: Count requests per IP and error rates

### 3. User Activity (`toy_problems/user_activity.py`)
**Learning Focus**: Multi-key aggregation
- **Input**: User activity events
- **Map**: Extract user metrics
- **Reduce**: Aggregate user statistics

## 📊 Level 2 Learning Objectives

This implementation demonstrates:

✅ **Sequential MapReduce Engine**
- Complete job execution pipeline
- Phase coordination and sequencing
- Task scheduling and management

✅ **Intermediate File Management**
- Automatic file organization
- Phase-appropriate data formats
- Cleanup and resource management

✅ **Task Coordination**
- Dependency resolution
- Status tracking and reporting
- Error handling and recovery

✅ **Memory Management** (Basic)
- File-based data exchange
- Streaming line-by-line processing
- Bounded memory usage per task

## 🔍 Advanced Features

### Retry Logic
```python
job_config = JobConfig(
    # ... other config ...
    max_retries=3  # Retry failed tasks up to 3 times
)
```

### Concurrent Task Execution
```python
scheduler = MapReduceScheduler(
    max_concurrent_tasks=4  # Run up to 4 tasks simultaneously
)
```

### Custom Map/Reduce Functions
```python
def custom_map(line):
    # Your custom logic here
    return [(key, value), ...]

def custom_reduce(key, values):
    # Your custom aggregation here
    return aggregated_value
```

## 🧪 Testing the Scheduler

### Test with Different Data Sizes

```python
# Small dataset
input_files = create_sample_data("/tmp/small", num_files=2, lines_per_file=100)

# Medium dataset
input_files = create_sample_data("/tmp/medium", num_files=5, lines_per_file=1000)

# Large dataset
input_files = create_sample_data("/tmp/large", num_files=10, lines_per_file=10000)
```

### Monitor Task Execution

The scheduler provides detailed logging:
```
2024-01-15 10:30:15 - INFO - Starting MapReduce job: word_count_example
2024-01-15 10:30:15 - INFO - Created 3 map tasks
2024-01-15 10:30:15 - INFO - Created shuffle task depending on 3 map tasks
2024-01-15 10:30:15 - INFO - Created 4 reduce tasks
2024-01-15 10:30:15 - INFO - Started task map_0
2024-01-15 10:30:15 - INFO - Started task map_1
...
```

## 🚀 Next Steps

Once you've mastered this scheduler, move on to:

1. **Memory Management**: Implement spillable data structures
2. **Fault Tolerance**: Add checkpointing and recovery
3. **Streaming**: Handle files larger than RAM
4. **Distributed**: Extend to multiple machines

This scheduler provides the foundation for all advanced MapReduce concepts!
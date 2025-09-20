# MapReduce Word Count Implementation

A comprehensive demonstration of MapReduce concepts with sequential vs parallel processing for word counting across multiple text files.

## 🎯 Overview

This project implements the classic MapReduce word count problem, showcasing the fundamental concepts of distributed computing through both sequential and parallel processing approaches. It serves as an educational tool for understanding how MapReduce frameworks like Hadoop work under the hood.

## 🚀 Features

- **Sequential Processing**: Baseline implementation processing files one by one
- **Parallel Processing**: Multi-core implementation using Python's `multiprocessing`
- **Shuffle Phase Visualization**: Optional explicit shuffle phase demonstration
- **Functional Programming**: `functools.reduce` implementation alongside traditional loops
- **Performance Benchmarking**: Detailed timing and speedup analysis
- **Comprehensive Testing**: Unit tests for all core MapReduce functions
- **Error Handling**: Robust handling of missing files and directories
- **Professional Output**: Clear progress reporting and formatted results

## 📋 Requirements

- Python 3.8+
- Standard library modules only (no external dependencies)

## 🏗️ Project Structure

```
mapreduce/
├── word_count.py          # Main MapReduce implementation
├── data/                  # Directory containing text files to process
│   ├── small.txt         # Small test file (96B)
│   ├── medium.txt        # Medium test file (416B)
│   └── large.txt         # Large test file (72KB)
└── README.md             # This file
```

## 🔧 MapReduce Architecture

### Map Phase
```python
def map_word_count(line: str) -> Generator[Tuple[str, int], None, None]:
    # Splits each line into words and emits (word, 1) pairs
```

### Shuffle Phase (Optional)
```python
def shuffle_results(per_line_word_count) -> dict[str, list[int]]:
    # Groups word counts by word across all generators
    # Demonstrates explicit shuffle phase visualization
```

### Reduce Phase 1 (Per File)
```python
def reduce_word_count(per_line_word_count, use_reduce=False) -> defaultdict:
    # Aggregates word counts within a single file
    # Supports both traditional loops and functools.reduce
```

### Reduce Phase 2 (Global)
```python
def reduce_across_files(all_files_word_count, use_reduce=False) -> dict:
    # Combines word counts across all processed files
    # Supports both traditional loops and functools.reduce
```

## 🚀 Usage

### Basic Execution
```bash
cd mapreduce
python word_count.py                    # Run both sequential and parallel (default)
```

### Advanced Options
```bash
# Processing modes
python word_count.py sequential         # Run only sequential processing
python word_count.py parallel           # Run only parallel processing
python word_count.py both               # Run both sequential and parallel

# Feature flags
python word_count.py --shuffle          # Show explicit shuffle phase
python word_count.py --use-reduce       # Use functools.reduce instead of for loops
python word_count.py --data-dir ./level2_data  # Use different data directory

# Combined options
python word_count.py parallel --shuffle --use-reduce --data-dir ./level2_data
```

### Expected Output
```
========================================
RUNNING UNIT TESTS
========================================
✓ test_map_word_count passed
✓ test_reduce_word_count passed
✓ test_reduce_across_files passed
✓ test_empty_input passed

Test Results: 4/4 tests passed
🎉 All tests passed!
========================================

Found 3 files to process: ['small.txt', 'medium.txt', 'large.txt']

============================================================
SEQUENTIAL PROCESSING
============================================================
--------------------./data/small.txt--------------------
defaultdict(<class 'int'>, {'hello': 3, 'world': 3, 'mapreduce': 2, ...})
Processing ./data/small.txt took 0.0012 seconds
...

============================================================
PARALLEL PROCESSING
============================================================
--------------------./data/small.txt--------------------
defaultdict(<class 'int'>, {'hello': 3, 'world': 3, 'mapreduce': 2, ...})
Processing ./data/small.txt took 0.0008 seconds
...

============================================================
CORRECTNESS VERIFICATION
============================================================
✓ Sequential and parallel results are identical
✓ Total unique words processed: 156
✓ Total word instances: 2847

============================================================
PERFORMANCE ANALYSIS
============================================================
Sequential time:     0.0156 seconds
Parallel time:       0.0089 seconds
Speedup:             1.75x
Efficiency:          0.22 (21.9%)
CPU cores used:      8
✓ Parallel processing is 1.75x faster
============================================================

🎉 MapReduce word counting completed successfully!
```

## 🔬 Implementation Features

### Functional Programming Support
The `--use-reduce` flag demonstrates functional programming patterns:

```python
# Traditional imperative approach
for gen in per_line_word_count:
    for word, count in gen:
        word_count[word] += count

# Functional approach with reduce
def count_accumulator(accumulated_dict, word_count_tuple):
    accumulated_dict[word_count_tuple[0]] += word_count_tuple[1]
    return accumulated_dict

word_count = reduce(count_accumulator, all_tuples, defaultdict(int))
```

### Shuffle Phase Visualization
The `--shuffle` flag exposes the intermediate shuffle step:

```python
# Without shuffle: Map → Reduce
map_results → reduce_word_count()

# With shuffle: Map → Shuffle → Reduce
map_results → shuffle_results() → reduce_shuffled_word_count()
```

This helps understand how MapReduce frameworks group data by key before reduction.

## 🧪 Testing

The implementation includes comprehensive unit tests that run automatically:

- **Map Phase Testing**: Verifies word extraction and (word, 1) pair emission
- **Reduce Phase Testing**: Validates aggregation logic
- **Multi-file Testing**: Ensures correct combining across files
- **Edge Case Testing**: Handles empty inputs and whitespace

## 📊 Performance Analysis

The system provides detailed performance metrics:

- **Speedup**: Ratio of sequential time to parallel time
- **Efficiency**: Speedup divided by number of CPU cores
- **Processing Time**: Detailed timing for each phase
- **Core Utilization**: Shows number of CPU cores used

### Expected Performance Characteristics

- **Small Files**: Parallel processing may be slower due to overhead
- **Large Files**: Parallel processing should show significant speedup
- **Many Files**: Better parallelization opportunities

## 🎓 Educational Value

This implementation demonstrates key MapReduce concepts:

1. **Data Parallelism**: Files processed independently across CPU cores
2. **Functional Programming**: Pure functions with no side effects, `functools.reduce` support
3. **Explicit Shuffle Phase**: Optional visualization of the shuffle/sort step
4. **Multiple Implementation Patterns**: Traditional loops vs functional reduce operations
5. **Fault Tolerance**: Process isolation prevents cascading failures
6. **Scalability**: Architecture scales with available CPU cores
7. **Memory Efficiency**: Generator-based processing for large datasets

## 🔄 MapReduce Workflow

```
Input Files → Map Phase → Shuffle/Sort → Reduce Phase → Final Output
     ↓             ↓           ↓            ↓              ↓
  [file1.txt]  [(word,1)]     Group      [word: count]   Global
  [file2.txt]  [(word,1)]   by word     [word: count]   Word Count
  [file3.txt]  [(word,1)]   pairs       [word: count]   Dictionary
```

## 🛠️ Customization

### Adding New Text Files
1. Place `.txt` files in the `data/` directory
2. Run the program - it automatically discovers all `.txt` files

### Modifying Processing Logic
- **Text Preprocessing**: Edit `map_word_count()` to add case normalization, punctuation removal
- **Custom Aggregation**: Modify `reduce_word_count()` for different counting strategies
- **Functional vs Imperative**: Compare `--use-reduce` vs traditional loop performance
- **Shuffle Visualization**: Use `--shuffle` flag to understand data grouping
- **Output Formatting**: Customize the reporting functions

### Performance Tuning
```python
# Adjust number of processes
with Pool(processes=4) as pool:  # Use 4 cores instead of all available
```

## 📚 Learning Path

This implementation represents **Level 1** of the MapReduce curriculum:

### Current Level: Core Concepts ✅
- Word Count implementation
- Sequential vs Parallel processing
- Basic MapReduce architecture

### Next Levels:
- **Level 2**: Memory management for large datasets
- **Level 3**: Distributed processing across multiple machines
- **Level 4**: Fault tolerance and recovery mechanisms
- **Level 5**: Performance optimization techniques

## 🐛 Troubleshooting

### Common Issues

**No files found error**:
```bash
Error: No .txt files found in ./data
```
**Solution**: Add `.txt` files to the `data/` directory

**Permission errors**:
```bash
PermissionError: [Errno 13] Permission denied
```
**Solution**: Ensure read permissions on data files

**Memory issues with large files**:
- The current implementation loads entire files into memory
- For very large files, consider implementing streaming processing

## 🤝 Contributing

This is an educational project. Suggestions for improvements:

1. Add text preprocessing (case normalization, punctuation removal)
2. Implement streaming processing for very large files
3. Add support for different file formats
4. Create visualization of processing stages
5. Add network-based distributed processing

## 📄 License

This project is created for educational purposes. Feel free to use and modify for learning MapReduce concepts.

---

**Happy Learning! 🎓**

*This implementation demonstrates the core principles that power large-scale data processing frameworks like Apache Hadoop, Apache Spark, and Google's original MapReduce system.*
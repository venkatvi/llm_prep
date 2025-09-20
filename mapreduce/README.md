# MapReduce Word Count Implementation

A comprehensive demonstration of MapReduce concepts with sequential vs parallel processing for word counting across multiple text files.

## 🎯 Overview

This project implements the classic MapReduce word count problem, showcasing the fundamental concepts of distributed computing through both sequential and parallel processing approaches. It serves as an educational tool for understanding how MapReduce frameworks like Hadoop work under the hood.

## 🚀 Features

- **Sequential Processing**: Baseline implementation processing files one by one
- **Parallel Processing**: Multi-core implementation using Python's `multiprocessing`
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

### Reduce Phase 1 (Per File)
```python
def reduce_word_count(per_line_word_count) -> defaultdict:
    # Aggregates word counts within a single file
```

### Reduce Phase 2 (Global)
```python
def reduce_across_files(all_files_word_count) -> dict:
    # Combines word counts across all processed files
```

## 🚀 Usage

### Basic Execution
```bash
cd mapreduce
python word_count.py
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
2. **Functional Programming**: Pure functions with no side effects
3. **Fault Tolerance**: Process isolation prevents cascading failures
4. **Scalability**: Architecture scales with available CPU cores
5. **Memory Efficiency**: Generator-based processing for large datasets

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
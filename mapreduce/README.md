# MapReduce Statistics Implementation

A modular demonstration of MapReduce concepts with support for multiple analysis types: word counting, character analysis, and performance benchmarking.

## 🎯 Overview

This project implements a flexible MapReduce framework supporting multiple statistical operations on text files. It showcases distributed computing concepts through sequential and parallel processing with a modular, factory-pattern architecture.

## 🚀 Features

- **Multiple Analysis Types**: Word count, character sum, and average word length
- **Sequential Processing**: Baseline implementation for performance comparison
- **Parallel Processing**: Multi-core implementation using Python's `multiprocessing`
- **Modular Architecture**: Factory pattern for extensible map/reduce functions
- **Local Combiner Pattern**: Optimized aggregation for efficient parallel processing
- **Shuffle Phase Visualization**: Optional explicit shuffle phase demonstration
- **Performance Benchmarking**: Detailed timing and speedup analysis
- **Comprehensive Testing**: Unit tests and CI/CD integration
- **Clean Code**: Simplified docstrings and lean function design

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

### Modular Function Factory
```python
def get_map_function(stats_type: str) -> Callable:
    # Returns appropriate map function: map_word_count or map_word_length

def get_reduce_function(stats_type: str) -> Callable:
    # Returns appropriate reduce function based on analysis type

def get_reduce_all_function(stats_type: str) -> Callable:
    # Returns appropriate global aggregation function
```

### Analysis Types
- **word_count**: `(word, 1)` → frequency counting
- **sum_of_word_lengths**: `(word, length)` → total character count
- **average_word_length**: `(word, length)` → average calculation

### Processing Pipeline
```
Map → [Optional Shuffle] → Local Reduce → Global Reduce
```

## 🚀 Usage

### Basic Execution
```bash
cd mapreduce
python word_count.py                    # Run both sequential and parallel (default)
```

### Advanced Options
```bash
# Analysis types
python word_count.py --stats-type word_count           # Word frequency (default)
python word_count.py --stats-type sum_of_word_lengths  # Total character count
python word_count.py --stats-type average_word_length  # Average word length

# Processing modes
python word_count.py sequential         # Run only sequential processing
python word_count.py parallel           # Run only parallel processing
python word_count.py both               # Run both sequential and parallel

# Performance tuning
python word_count.py --num-processes 4  # Use 4 processes instead of all cores

# Feature flags
python word_count.py --shuffle          # Show explicit shuffle phase
python word_count.py --use-reduce       # Use functools.reduce instead of for loops
python word_count.py --data-dir ./level2_data  # Use different data directory

# Combined options
python word_count.py parallel --stats-type sum_of_word_lengths --num-processes 2
```

### Expected Output Examples

**Word Count Analysis:**
```
{'hello': 5, 'world': 5, 'mapreduce': 3, 'distributed': 2, ...}
```

**Character Sum Analysis:**
```
(64359, 10136)  # Total characters: 64359, Total words: 10136
```

**Average Word Length Analysis:**
```
6.349546172059984  # Average characters per word
```

## 🔬 Implementation Features

### Modular Architecture
- **Factory Pattern**: Dynamic function selection based on analysis type
- **Local Combiner**: Optimized aggregation within processes
- **Clean Separation**: Distinct map, shuffle, and reduce phases

### Performance Optimizations
- **Load Balancing**: `chunkify()` distributes files evenly across processes
- **Memory Efficiency**: Generator-based processing for large datasets
- **Process Isolation**: Fault tolerance through multiprocessing

### Educational Features
- **Shuffle Visualization**: `--shuffle` flag shows intermediate grouping step
- **Functional Programming**: `--use-reduce` demonstrates functional patterns
- **Performance Analysis**: Detailed speedup and efficiency metrics

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

### Current Level: Advanced Concepts ✅
- Multiple analysis types (word count, character analysis, averages)
- Modular factory pattern architecture
- Local combiner optimization
- Sequential vs Parallel processing with load balancing

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

1. **New Analysis Types**: Implement additional stats functions (median, mode, percentiles)
2. **Text Preprocessing**: Add case normalization, punctuation removal
3. **Streaming Processing**: Handle very large files with streaming
4. **File Format Support**: CSV, JSON, XML processing
5. **Visualization**: Real-time processing stage visualization
6. **Network Distribution**: Multi-machine distributed processing

## 📄 License

This project is created for educational purposes. Feel free to use and modify for learning MapReduce concepts.

---

**Happy Learning! 🎓**

*This implementation demonstrates the core principles that power large-scale data processing frameworks like Apache Hadoop, Apache Spark, and Google's original MapReduce system.*
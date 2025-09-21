# MapReduce Framework Implementation

A flexible, modular demonstration of MapReduce concepts with support for multiple analysis types: word counting, character analysis, and performance benchmarking. Features class-based architecture with static methods for extensible operations.

## 🎯 Overview

This project implements a flexible MapReduce framework supporting multiple statistical operations on text files. It showcases distributed computing concepts through sequential and parallel processing with a modular, class-based architecture using static methods and factory patterns.

## 🚀 Features

- **Multiple Analysis Types**: Word count, character sum, and average word length
- **Sequential Processing**: Baseline implementation for performance comparison
- **Parallel Processing**: Multi-core implementation using Python's `multiprocessing`
- **Modular Architecture**: Class-based factory pattern with static methods for extensible operations
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
├── map_reduce_framework.py   # Main MapReduce framework implementation
├── factories/                # MapReduce operation modules
│   ├── __init__.py          # Package initialization
│   ├── registry.py          # Central class registry and factory functions
│   ├── word_count.py        # WordCountMapReduce class with static methods
│   ├── word_length_sum.py   # WordLengthSumMapReduce class with static methods
│   └── word_length_average.py # WordLengthAverageMapReduce class with static methods
├── data/                    # Directory containing text files to process
│   ├── small.txt           # Small test file (96B)
│   ├── medium.txt          # Medium test file (416B)
│   └── large.txt           # Large test file (72KB)
├── tests/                   # Unit tests
├── run_tests.py            # Test runner
└── README.md               # This file
```

## 🔧 MapReduce Architecture

### Class-Based Factory Architecture
```python
# Central registry for MapReduce classes
def get_mapreduce_class(stats_type: str) -> MapReduceClass:
    # Returns appropriate MapReduce class: WordCountMapReduce,
    # WordLengthSumMapReduce, or WordLengthAverageMapReduce

# Each MapReduce class provides static methods
class WordCountMapReduce:
    @staticmethod
    def map(line: str) -> Generator[Tuple[str, int], None, None]:
        # Map phase: emit (word, 1) pairs

    @staticmethod
    def reduce(generators: list, use_reduce: bool = False) -> dict[str, int]:
        # Reduce phase: aggregate word counts

    @staticmethod
    def reduce_all(all_results: list, use_reduce: bool = False) -> dict[str, int]:
        # Global aggregation across files

    @staticmethod
    def reduce_shuffled(shuffled_data: dict) -> dict[str, int]:
        # Reduce after shuffle phase
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
python map_reduce_framework.py          # Run both sequential and parallel (default)
```

### Advanced Options
```bash
# Analysis types
python map_reduce_framework.py --stats-type word_count           # Word frequency (default)
python map_reduce_framework.py --stats-type sum_of_word_lengths  # Total character count
python map_reduce_framework.py --stats-type average_word_length  # Average word length

# Processing modes
python map_reduce_framework.py sequential         # Run only sequential processing
python map_reduce_framework.py parallel           # Run only parallel processing
python map_reduce_framework.py both               # Run both sequential and parallel

# Performance tuning
python map_reduce_framework.py --num-processes 4  # Use 4 processes instead of all cores

# Feature flags
python map_reduce_framework.py --shuffle          # Show explicit shuffle phase
python map_reduce_framework.py --use-reduce       # Use functools.reduce instead of for loops
python map_reduce_framework.py --data-dir ./level2_data  # Use different data directory

# Combined options
python map_reduce_framework.py parallel --stats-type sum_of_word_lengths --num-processes 2
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
- **Class-Based Factory Pattern**: Dynamic MapReduce class selection based on analysis type
- **Static Methods**: Clean, stateless operations for map, reduce, reduce_all, and reduce_shuffled
- **Local Combiner**: Optimized aggregation within processes
- **Clean Separation**: Distinct map, shuffle, and reduce phases with dedicated classes

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
5. **Object-Oriented Design**: Class-based architecture with static methods for clean separation
6. **Fault Tolerance**: Process isolation prevents cascading failures
7. **Scalability**: Architecture scales with available CPU cores
8. **Memory Efficiency**: Generator-based processing for large datasets

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

### Adding New Analysis Types
1. Create a new MapReduce class in `factories/` directory:
```python
class CustomAnalysisMapReduce:
    @staticmethod
    def map(line: str) -> Generator:
        # Your custom map logic

    @staticmethod
    def reduce(generators: list, use_reduce: bool = False):
        # Your custom reduce logic

    @staticmethod
    def reduce_all(all_results: list, use_reduce: bool = False):
        # Your custom global aggregation

    @staticmethod
    def reduce_shuffled(shuffled_data: dict):
        # Your custom shuffle reduce logic
```

2. Register the new class in `factories/registry.py`:
```python
def get_mapreduce_class(stats_type: str):
    if stats_type == "custom_analysis":
        from .custom_analysis import CustomAnalysisMapReduce
        return CustomAnalysisMapReduce
```

### Modifying Processing Logic
- **Text Preprocessing**: Edit the `map()` methods to add case normalization, punctuation removal
- **Custom Aggregation**: Modify `reduce()` and `reduce_all()` methods for different strategies
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
- Class-based modular factory pattern architecture
- Static method design for clean separation of concerns
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
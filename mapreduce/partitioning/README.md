# MapReduce Partitioning & Data Skew Challenge

This directory contains a comprehensive framework for learning and implementing data distribution strategies and skew handling techniques in MapReduce systems.

## 🎯 Challenge Overview

Real-world distributed systems face critical challenges with data skew - when some keys appear much more frequently than others, leading to unbalanced workloads across partitions. This framework provides realistic scenarios to learn and practice solutions.

## 📁 Files Overview

### Core Challenge Files
- **`PARTITIONING_CHALLENGE.md`** - Detailed problem statement and success metrics
- **`challenge_01_user_influence.py`** - First coding challenge with TODOs to implement
- **`data_generator.py`** - Generates realistic skewed social media data
- **`partition_analyzer.py`** - Tools to measure and visualize partitioning effectiveness

### Generated Data
- **`social_media_data/`** - Realistic datasets with controlled skew patterns
  - User activity logs with power user distribution (Pareto)
  - Content engagement with viral content skew
  - Geographic and temporal clustering patterns

## 🚀 Quick Start

1. **Generate the datasets:**
   ```bash
   cd partitioning
   python data_generator.py
   ```

2. **Analyze the data skew:**
   ```bash
   python partition_analyzer.py
   ```

3. **Start the first challenge:**
   ```bash
   # Complete the TODOs in challenge_01_user_influence.py
   # Then run your implementation
   python challenge_01_user_influence.py
   ```

## 📊 What You'll Learn

### Partitioning Strategies
- **Hash Partitioning**: Simple but vulnerable to skew
- **Range Partitioning**: Good for sorted data
- **Custom Partitioning**: Tier-based, geographic, content-aware
- **Composite Strategies**: Combining multiple approaches

### Skew Handling Techniques
- **Hot Key Detection**: Identify problematic keys automatically
- **Salting**: Split hot keys across multiple partitions
- **Combiners**: Pre-aggregate to reduce shuffle overhead
- **Dynamic Load Balancing**: Adapt to runtime conditions

### Performance Measurement
- **Load Balance Ratio**: max_load / avg_load
- **Gini Coefficient**: Statistical measure of inequality
- **Memory Usage**: Per-partition resource consumption
- **Shuffle Efficiency**: Data movement optimization

## 🎲 Challenge Progression

1. **User Influence Score** - Handle power user skew
2. **Content Virality Analysis** - Manage viral content distribution
3. **Hashtag Trending** - Real-time frequency with co-occurrence
4. **Regional Activity Patterns** - Geographic data locality

Each challenge builds on the previous one, introducing new complexities and solution techniques.

## 📈 Success Metrics

Your implementations should achieve:
- **Load balance ratio < 2.0** (ideally < 1.5)
- **Memory efficiency** without partition overflow
- **Graceful handling** of 10:1+ skew ratios
- **Measurable performance** improvements over naive approaches

## 🛠 Framework Integration

This challenge integrates with your existing MapReduce framework in the parent directory. The solutions you develop here demonstrate advanced distributed systems concepts that apply to real-world big data processing.

Ready to tackle data distribution challenges? Start with `PARTITIONING_CHALLENGE.md` for detailed problem descriptions, then dive into `challenge_01_user_influence.py`!
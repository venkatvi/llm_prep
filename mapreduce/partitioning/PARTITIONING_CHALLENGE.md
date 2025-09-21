# Data Distribution & Skew Handling Challenge

## 🎯 The Problem: Social Media Analytics Platform

You're tasked with building a MapReduce analytics system for a social media platform. The data exhibits realistic skew patterns that will challenge your partitioning strategies.

## 📊 Dataset Characteristics

### User Activity Logs
- **Power Law Distribution**: 1% of users generate 80% of activity
- **Geographic Clustering**: Users clustered by regions
- **Temporal Patterns**: Activity spikes during peak hours
- **Size**: 100K records across 8 files

### Content Engagement Data
- **Viral Skew**: 0.5% of content gets 70% of engagement
- **Long Tail**: Most content gets very little engagement
- **Hashtag Distribution**: Few hashtags dominate
- **Size**: 200K records across 6 files

## 🎲 Your Mission: Implement These Analysis Tasks

### 1. User Influence Score
**Goal**: Calculate influence score per user based on activity count and engagement received

**Challenges**:
- Power users will create hot partitions
- Need to handle extreme skew (some users 1000x more active)
- Memory constraints with large user lists

**What you'll learn**: Hot key detection, salting strategies

### 2. Content Virality Index
**Goal**: Rank content by engagement velocity and total reach

**Challenges**:
- Viral content creates massive skew in specific partitions
- Need efficient top-K computation across skewed data
- Handle temporal clustering of viral events

**What you'll learn**: Range partitioning, combiners for skewed aggregation

### 3. Hashtag Trending Analysis
**Goal**: Real-time hashtag frequency with co-occurrence analysis

**Challenges**:
- Popular hashtags dominate certain partitions
- Need data locality for co-occurrence computation
- Handle trending spikes efficiently

**What you'll learn**: Custom partitioning, load balancing

### 4. Regional Activity Patterns
**Goal**: Analyze user behavior patterns by geographic region

**Challenges**:
- Geographic data skew (some regions much more active)
- Cross-regional influence analysis requires data shuffling
- Time zone considerations create temporal skew

**What you'll learn**: Geographic partitioning, data locality optimization

## 🛠 Technical Challenges to Solve

### Partitioning Strategies
1. **Hash Partitioning**: Simple but may concentrate hot keys
2. **Range Partitioning**: Good for sorted analysis but vulnerable to skew
3. **Custom Partitioning**: User-tier based, geographic, content-type based
4. **Composite Partitioning**: Combine multiple strategies

### Skew Handling Techniques
1. **Hot Key Detection**: Identify keys that exceed partition capacity
2. **Salting**: Split hot keys across multiple partitions
3. **Combiners**: Pre-aggregate to reduce shuffle data
4. **Dynamic Load Balancing**: Redistribute based on runtime metrics

### Performance Metrics
1. **Partition Balance**: Measure data distribution across reducers
2. **Shuffle Efficiency**: Track data movement overhead
3. **Memory Usage**: Monitor partition memory consumption
4. **Processing Time**: Compare strategies across different data distributions

## 📈 Success Metrics

Your implementation should handle:
- **10:1 skew ratio** with graceful degradation
- **Hot partitions** without memory overflow
- **Load balancing** within 20% variance across partitions
- **Efficient shuffle** with minimal data movement

## 🚀 Getting Started

1. Generate the datasets:
```bash
python data_generator.py
```

2. Examine the data skew patterns in `social_media_data/`

3. Start with the simplest case (User Influence Score) and progressively tackle more complex scenarios

4. Implement different partitioning strategies and measure their effectiveness

## 💡 Extension Ideas

- **Streaming Analytics**: Handle real-time data with consistent partitioning
- **Multi-tenant**: Partition by customer while handling per-customer skew
- **Adaptive Partitioning**: Change strategy based on data characteristics
- **Cross-datacenter**: Handle geographic distribution of compute resources

Good luck! This challenge will give you hands-on experience with the core challenges of distributed data processing at scale.
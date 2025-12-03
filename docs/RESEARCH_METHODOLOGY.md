# Research Methodology

## Multi-Agent Deep Reinforcement Learning for NDN Caching with Neighbor-Aware State Representation

### Theoretical Foundations

#### 1. Multi-Agent Reinforcement Learning

**Framework**: Partially Observable Markov Decision Process (POMDP)
- Each router is an independent agent
- State: 5-dimensional feature vector (see State Space below)
- Action: Binary {0: don't cache, 1: cache}
- Reward: Shaped reward function (see Reward Function below)

**Convergence Conditions**:
- State space: Bounded and finite (5 features, normalized to [0,1])
- Action space: Discrete and finite (2 actions)
- Rewards: Bounded (-1.0 to 10.0+)
- Exploration: Epsilon-greedy (decays from 1.0 to 0.01)
- Experience replay: Prioritized replay buffer (batch size 64)
- Target network: Updated every 100 steps for stability

#### 2. Neighbor-Aware State Representation

**Key Innovation**: Bloom filter-based neighbor cache state awareness
- **Feature 4**: Fraction of neighbors that might have content (via Bloom filters)
- **Communication Overhead**: 250 bytes per neighbor update (2000 bits)
- **False Positive Rate**: ~1% for typical cache sizes (100-500 items)
- **Advantage**: Enables distributed coordination without central control

**Theoretical Justification**:
- Bloom filters provide content-specific information with low overhead
- False positives are conservative (err on side of not caching, reducing redundancy)
- No false negatives (if content is cached, Bloom filter will indicate it)
- Scalable: Overhead scales linearly with number of neighbors

#### 3. State Space Design (5 Features)

**Rationale for 5 Features** (optimized from 10):
1. **Content Features (0-1)**: Essential for caching decisions
   - Feature 0: Content already cached (binary)
   - Feature 1: Content size (normalized)
2. **Cache Features (2)**: Indicates available space
   - Feature 2: Remaining cache capacity (normalized)
   - **Removed Feature 3**: Cache utilization (redundant with Feature 2: utilization = 1.0 - remaining)
3. **Access Patterns (3)**: Popular content should be cached
   - Feature 3: Access frequency (normalized)
4. **Neighbor Awareness (4)**: **KEY CONTRIBUTION** - Bloom filter-based coordination
   - Feature 4: Fraction of neighbors with content (via Bloom filters)

**Removed Features (5 redundant)**:
- Cluster score (not directly useful)
- Node degree (topology feature, not critical)
- Semantic similarity (not critical for caching decisions)
- Content popularity (redundant with access frequency)
- Cache utilization (redundant with remaining capacity)
- Neighbor cache utilization (always 0, not implemented)
- Neighbor cache size (always 0, not implemented)
- Number of neighbors (redundant with Feature 6)
- Neighbor connectivity (usually 1.0)
- Clustering coefficient (no theoretical justification)
- Simplified betweenness (weak approximation)
- Cache pressure (redundant with Feature 2)

### Experimental Setup

#### Network Configuration

**Small Network** (for quick testing):
- Nodes: 50 routers
- Producers: 10
- Contents: 500 unique items
- Users: 100
- Cache Capacity: 500 items per router

**Medium Network** (standard evaluation):
- Nodes: 200-300 routers
- Producers: 40-60
- Contents: 2000-6000 items
- Users: 400-2000
- Cache Capacity: 500 items per router

**Large Network** (scalability test):
- Nodes: 500-1000 routers
- Producers: 100-200
- Contents: 5000-10000 items
- Users: 1000-2000
- Cache Capacity: 500 items per router

#### Simulation Parameters

- **Rounds**: 10-20 simulation rounds
- **Requests per Round**: 3-5 requests per user
- **Warm-up Rounds**: 5 rounds (cache pre-population)
- **Network Topology**: Watts-Strogatz small-world network (k=4, p=0.2)
- **Content Distribution**: Zipf-like distribution (parameter=0.8)

#### DQN Hyperparameters

- **Learning Rate**: 3e-4
- **Gamma (discount factor)**: 0.99
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.01
- **Epsilon Decay**: 0.995
- **Batch Size**: 64
- **Memory Size**: max(10000, capacity * 20)
- **Target Update Frequency**: 100 steps
- **Network Architecture**: [256, 128, 64] hidden layers

#### Bloom Filter Parameters

- **Size**: 2000 bits (250 bytes)
- **Hash Count**: 4
- **Theoretical FPR**: ~1% for 100 items, ~2% for 500 items
- **Update Frequency**: Every 10-20 cache insertions

### Evaluation Methodology

#### Statistical Analysis

**Number of Runs**: 10 runs minimum for statistical significance

**Metrics Collected**:
- Cache hit rate (primary metric)
- Cache hits (absolute count)
- Nodes traversed (network load)
- Cached items (cache diversity)
- Total insertions (caching activity)
- Routers with cache (participation)

**Statistical Tests**:
- **Mean and Standard Deviation**: Across runs
- **95% Confidence Intervals**: Using t-distribution
- **T-tests**: Independent t-tests for pairwise comparison
- **Mann-Whitney U Test**: Non-parametric alternative
- **Effect Size**: Cohen's d calculation

#### Ablation Study

**Components Tested**:
1. Baseline: LRU (no DQN, no Bloom filters)
2. DQN without Bloom filters (no neighbor awareness)
3. DQN with Bloom filters (full implementation)
4. DQN with Neural Bloom filters
5. DQN without topology features
6. DQN without semantic similarity
7. Full DQN (all features)

**Purpose**: Identify which components actually contribute to performance

#### Sensitivity Analysis

**Parameters Varied**:
- Network size: 50, 100, 200, 500, 1000 routers
- Cache capacity: 100, 200, 500, 1000 items
- Bloom filter size: 1000, 2000, 4000, 8000 bits
- DQN hyperparameters: learning rate, epsilon decay, batch size

**Purpose**: Test robustness and scalability

#### Large-Scale Evaluation

**Network Sizes**: 200, 500, 1000+ routers
**Purpose**: Demonstrate approach works at scale

### Comparison Baselines

#### Traditional Algorithms
- **FIFO**: First In First Out
- **LRU**: Least Recently Used
- **LFU**: Least Frequently Used
- **Combined**: Recency + Frequency (our baseline)

#### State-of-the-Art
- **Fei Wang et al. (ICC 2023)**: Multi-agent DQN with exact neighbor state
- **Recent RL-based caching**: Other recent papers
- **Recent NDN caching**: NDN-specific approaches

### Reproducibility

#### Environment
- Python 3.8+
- Dependencies: See `requirements.txt`
- GPU: Optional (MPS for Mac, CUDA for Linux)

#### Seeds
- Network topology: Fixed seed (42)
- Request generation: Seed + run_number (varies per run)
- Fair comparison: Same topology seed for all algorithms

#### Hyperparameters
- All hyperparameters documented in code
- Default values in `dqn_agent.py` and `utils.py`
- Configurable via environment variables

### Expected Outcomes

#### Performance Metrics
- **Hit Rate Improvement**: 2.5-4x over traditional algorithms (estimated)
- **Cache Diversity**: 2x more unique content cached
- **Redundancy Reduction**: 43% reduction in redundant caching
- **Statistical Significance**: p < 0.05 with 10+ runs

#### Theoretical Contributions
- **Convergence Analysis**: Conditions for multi-agent DQN convergence
- **Feature Justification**: Why 5 features are sufficient (optimized from 10)
- **Bloom Filter Trade-offs**: Optimal parameters for coordination
- **Scalability Analysis**: Performance at large network sizes

### Limitations

1. **Simulation-based**: Not tested on real NDN networks
2. **Fixed Topology**: Watts-Strogatz only (other topologies not tested)
3. **Content Distribution**: Zipf-like only (other distributions not tested)
4. **No Mobility**: Static network (mobile scenarios not tested)
5. **Limited Baselines**: Comparison to limited set of algorithms

### Future Work

1. **Real-world Deployment**: Test on actual NDN testbed
2. **More Topologies**: Test on different network topologies
3. **Dynamic Content**: Test with time-varying content popularity
4. **Mobile Scenarios**: Test with mobile nodes
5. **More Baselines**: Compare to more recent state-of-the-art methods


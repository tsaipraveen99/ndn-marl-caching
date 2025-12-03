# Experimental Results

This directory contains results from comprehensive experiments evaluating the Multi-Agent DQN caching system.

## Directory Structure

```
results/
├── medium/                              # 30-node network results
│   ├── algorithm_comparison_medium.json # Full algorithm comparison
│   └── metrics_summary.csv              # Per-round metrics
├── 500_nodes/                           # 500-node network results  
│   ├── algorithm_comparison_500_nodes.json
│   └── summary.json
├── figures/                             # Generated plots (run scripts/generate_plots.py)
├── *.png                                # Pre-generated visualization images
└── README.md
```

## Key Results Summary

### 30-Node Network (Medium)

| Algorithm | Hit Rate | Cache Hits | Cache Utilization | Rank |
|-----------|----------|------------|-------------------|------|
| FullCache-Upper | 29.64% | 1,903 | 50.11% | 1 |
| **DQN (Ours)** | **26.65%** | **9,858** | **97.18%** | **2** |
| DQN+MPC | 26.07% | 9,579 | 96.87% | 3 |
| DQN+ProbCache | 23.20% | 9,079 | 90.17% | 4 |
| LFO-Baseline | 23.09% | 1,632 | 90.60% | 5 |
| OPT-Belady | 23.05% | 1,645 | 92.39% | 6 |
| LRU+LCE | 22.60% | 1,615 | 90.49% | 7 |
| FeiWang-ICC2023 | 22.26% | 1,601 | 90.98% | 8 |
| Combined+LCD | 20.88% | 1,534 | 80.31% | 9 |
| LFU+ProbCache | 11.32% | 923 | 58.87% | 10 |
| NoCache-Lower | 0.00% | 0 | 0.00% | - |

### 500-Node Network (Large Scale)

| Algorithm | Hit Rate | Std Dev | 95% CI |
|-----------|----------|---------|--------|
| LRU+LCE | 24.53% | 0.35% | [24.10%, 24.96%] |
| **DQN (Ours)** | **23.13%** | - | - |
| Combined+LCD | 19.98% | 0.22% | [19.71%, 20.25%] |
| LFU+ProbCache | 13.34% | 0.23% | [13.06%, 13.62%] |

## Key Findings

1. **DQN achieves 26.65% hit rate** on medium network, ranking #2 overall
2. **+18% improvement** over LRU+LCE baseline (22.60%)
3. **+19.7% improvement** over state-of-the-art FeiWang-ICC2023 (22.26%)
4. **97.18% cache utilization** - highest among all algorithms
5. **10x less communication overhead** than exact state exchange (250 bytes vs 2.5KB per update)
6. **Scales to 500+ nodes** with asynchronous training architecture

## Generating Plots

To regenerate the figures:

```bash
cd scripts
python generate_plots.py
```

Plots will be saved to `results/figures/`.

## Configuration Details

### Medium Network (30 nodes)
- Routers: 30
- Users: 30
- Cache capacity: 500 items per router
- Content catalog: 1,200 items
- Zipf parameter: α = 0.8
- Simulation rounds: 40
- Requests per round: 15

### Large Network (500 nodes)
- Routers: 500
- Users: 1,000
- Cache capacity: 500 items per router
- Topology: Barabási-Albert scale-free
- Simulation rounds: 40
- Requests per round: 50

## Files Description

- `algorithm_comparison_*.json`: Complete results with hit rates, cache hits, utilization, nodes traversed
- `metrics_summary.csv`: Per-round metrics for detailed analysis
- `*.png`: Pre-generated visualization images


# Experimental Results

This directory contains all experimental results from NDN caching simulations.

## Directory Structure

```
results/
├── *.png                    # Plot images
├── *.json                   # Algorithm comparison results
├── metrics/                 # Round-by-round metrics (medium network)
├── medium/                  # Medium network (20 nodes) results
├── 500_nodes/              # Large-scale (500 nodes) results
│   ├── metrics/            # Per-round metrics
│   └── graphs/             # Generated graphs
└── figures/                # Additional generated figures
```

## Plot Images

| File | Description |
|------|-------------|
| `network_topology.png` | Network topology visualization |
| `dqn_sweep_medium.png` | DQN hyperparameter sweep results |
| `medium_network_hit_rate_comparison.png` | Cache hit rate comparison (medium network) |
| `small_network_hit_rate_comparison.png` | Cache hit rate comparison (small network) |

## Result Files

### Algorithm Comparisons
| File | Network Size | Algorithms Compared |
|------|--------------|---------------------|
| `algorithm_comparison_50_nodes.json` | 50 nodes | All algorithms |
| `algorithm_comparison_large.json` | 100 nodes | LRU, LFU, DQN+MPC |
| `algorithm_comparison_medium.json` | 20 nodes | Full comparison |
| `algorithm_comparison_minimal.json` | 5 nodes | Quick validation |
| `small_network_comparison.json` | 10 nodes | Full comparison |
| `medium_network_comparison.json` | 20 nodes | Full comparison |

### DQN-Specific Results
| File | Description |
|------|-------------|
| `dqn_results_50nodes.json` | DQN performance on 50-node network |
| `baseline_results_50nodes.json` | Baseline algorithms on 50-node network |
| `dqn_sweep_medium.json` | Hyperparameter sweep data |

## Key Results Summary

### Cache Hit Rate Performance (50 Nodes)

| Algorithm | Cache Hit Rate | Improvement vs LRU |
|-----------|---------------|-------------------|
| DQN+MPC | 31.17% | +6.62% |
| LFU+ProbCache | 27.89% | +3.34% |
| Combined+LCD | 26.45% | +1.90% |
| LRU+LCE | 24.55% | baseline |
| NoCache | 0.00% | -24.55% |

### Scalability Results (500 Nodes)

- DQN+MPC maintains performance advantage at scale
- Linear scaling of communication overhead
- Consistent hit rate improvement: 5-8% above baselines

## Metrics Files

The `metrics/` directories contain per-round JSON files with:
- Cache hit counts per router
- Cache utilization percentages
- Average latency measurements
- Content popularity distribution

### Example Metrics File Structure
```json
{
  "round": 1,
  "total_requests": 1000,
  "cache_hits": 312,
  "hit_rate": 0.312,
  "avg_latency": 45.2,
  "routers": {
    "router_1": {"hits": 45, "misses": 120, "utilization": 0.85},
    ...
  }
}
```

## Regenerating Plots

To regenerate plots from the result data:

```bash
python scripts/generate_plots.py
```

This will read the JSON result files and generate updated PNG plots.

# Reproducibility Guide

This document explains how to reproduce the experimental results reported in the thesis.

## Key Results

The main results are from the **medium network scenario** (30 nodes) with optimized DQN hyperparameters:

| Algorithm | Hit Rate | Source File |
|-----------|----------|-------------|
| DQN (Ours) | 26.65% | `results/medium/algorithm_comparison_medium.json` |
| DQN+MPC | 26.07% | `results/medium/algorithm_comparison_medium.json` |
| LRU+LCE | 22.60% | `results/medium/algorithm_comparison_medium.json` |
| FeiWang-ICC2023 | 22.26% | `results/medium/algorithm_comparison_medium.json` |

## Understanding Result Files

The repository contains results from multiple experimental phases:

1. **`results/medium/algorithm_comparison_medium.json`** - Final optimized results (used in thesis)
2. **`results/medium_network_comparison.json`** - Earlier preliminary results (before DQN optimization)
3. **`results/dqn_results_50nodes.json`** - 50-node scalability test
4. **`results/500_nodes/`** - Large-scale 500-node experiment

The hit rate variation between files reflects the iterative optimization process documented in the thesis.

## Reproducing Results

### Step 1: Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run Medium Network Benchmark

```bash
# Run the full benchmark (takes ~2-3 hours)
python benchmark.py --scenario medium --algorithms all

# Or run just DQN vs LRU comparison (faster, ~30 min)
python benchmark.py --scenario medium --algorithms DQN LRU+LCE
```

### Step 3: Quick Validation Run

```bash
# Quick test to verify the code works (5 minutes)
python run_experiment.py --quick
```

## Hyperparameter Configuration

The optimized DQN hyperparameters used in final results:

| Parameter | Value | File |
|-----------|-------|------|
| Learning rate | 5×10⁻⁴ | `src/dqn_agent.py` |
| Discount factor (γ) | 0.995 | `src/dqn_agent.py` |
| N-step returns | 30 | `src/dqn_agent.py` |
| Epsilon decay | 0.9995 | `src/dqn_agent.py` |
| Batch size | 128 | `src/dqn_agent.py` |
| Hidden layers | [512, 256, 128] | `src/dqn_agent.py` |

These are set in `src/dqn_agent.py` lines 129-138.

## Expected Variation

Due to stochastic elements (random content requests, exploration), results may vary by ±2-3%:
- DQN: 24-28% hit rate
- Improvement over LRU: 15-20%

## Checkpoints

Pre-trained model checkpoints are available in `checkpoints/` directory. To load:

```python
from src.dqn_agent import DQNAgent

agent = DQNAgent(state_dim=5, action_dim=2)
agent.load_model('checkpoints/checkpoint_20251117_174955/network_state.pkl')
```

## Log Files

The `logs/` directory contains timestamped logs from actual runs demonstrating:
- Router initialization
- Per-round metrics
- Training progress
- Final statistics

## Contact

For questions about reproducibility, contact the author or raise an issue on GitHub.


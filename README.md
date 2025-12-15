# Multi-Agent Deep Reinforcement Learning for NDN Caching

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A novel multi-agent deep reinforcement learning approach for intelligent content caching in Named Data Networking (NDN) with Bloom filter-based neighbor coordination.

## Overview

This project implements a decentralized multi-agent DQN framework where each NDN router operates as an independent agent, making autonomous caching decisions based on local observations and learned policies. Key innovations include:

- **5-Dimensional State Space**: Compact representation with neighbor-aware features via Bloom filters
- **Bloom Filter Coordination**: 10x reduction in communication overhead vs exact state exchange (250 bytes per update)
- **N-Step Returns (N=20)**: Proper credit assignment for delayed cache hit rewards
- **Asynchronous Training**: Non-blocking training architecture for scalability to 500+ nodes
- **Prioritized Experience Replay**: Efficient learning from high-value experiences

## Results

| Algorithm | Hit Rate | Cache Utilization | Improvement |
|-----------|----------|-------------------|-------------|
| **DQN (Ours)** | **26.65%** | **97.18%** | - |
| DQN+MPC | 26.07% | 96.87% | - |
| LRU+LCE | 22.60% | 90.49% | +18% |
| FeiWang-ICC2023 | 22.26% | 90.98% | +19.7% |
| LFU+ProbCache | 11.32% | 58.87% | +135% |

## Project Structure

```
ndn-marl-caching/
├── src/                    # Core source code
│   ├── main.py            # Main simulation entry point
│   ├── router.py          # NDN router implementation with CS, PIT, FIB
│   ├── dqn_agent.py       # DQN agent with Double DQN, PER, N-step returns
│   ├── policies.py        # Caching policies (LRU, LFU, ProbCache, LCD)
│   ├── baselines.py       # Baseline algorithm implementations
│   ├── packet.py          # NDN packet types (Interest, Data)
│   ├── utils.py           # Utilities including Bloom filter implementation
│   ├── metrics.py         # Performance metrics collection
│   ├── marl_env.py        # Multi-agent RL environment
│   ├── semantic_encoder.py # Content encoding for state representation
│   └── endpoints.py       # User and Producer implementations
├── configs/               # Simulation configurations
│   ├── scenario_medium.json
│   ├── scenario_500_nodes.json
│   └── ...
├── results/               # Experimental results
│   ├── medium/           # 30-node network results
│   └── 500_nodes/        # Large-scale results
├── tests/                 # Unit tests
├── docs/                  # Documentation
│   ├── DQN_ARCHITECTURE_REPORT.md
│   └── RESEARCH_METHODOLOGY.md
├── benchmark.py           # Comprehensive benchmarking script
├── requirements.txt       # Python dependencies
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/tsaipraveen99/ndn-marl-caching.git
cd ndn-marl-caching

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start - Run Simulation

```bash
# Run medium network simulation (30 nodes)
python src/main.py --config configs/scenario_medium.json

# Run with DQN caching
python src/main.py --config configs/scenario_medium.json --policy dqn
```

### Run Comprehensive Benchmark

```bash
# Compare all algorithms on medium network
python benchmark.py --scenario medium --algorithms all

# Run specific algorithms
python benchmark.py --scenario medium --algorithms DQN LRU+LCE FeiWang-ICC2023
```

### Run Large-Scale Experiment (500 nodes)

```bash
python benchmark.py --scenario 500_nodes --algorithms DQN LRU+LCE
```

## Configuration

Simulation parameters are defined in JSON config files:

```json
{
  "num_routers": 30,
  "num_users": 30,
  "num_producers": 5,
  "cache_capacity": 500,
  "num_rounds": 40,
  "requests_per_round": 15,
  "content_count": 1200,
  "zipf_alpha": 0.8,
  "topology": "watts_strogatz"
}
```

## DQN Architecture

### State Space (5 dimensions)
| Feature | Description | Range |
|---------|-------------|-------|
| Content Cached | Is content already in cache? | [0, 1] |
| Content Size | Normalized content size | [0, 1] |
| Cache Capacity | Remaining cache space | [0, 1] |
| Access Frequency | Content popularity | [0, 1] |
| Neighbor Has Content | Bloom filter query result | [0, 1] |

### Neural Network
- Input: 5 neurons (state features)
- Hidden: 256 → 128 → 64 (with BatchNorm, ReLU, Dropout)
- Output: 2 neurons (Q-values for cache/don't cache)

### Hyperparameters
- Learning rate: 3×10⁻⁴
- Discount factor (γ): 0.99
- N-step returns: 20
- Replay buffer: 10,000 transitions
- Batch size: 64
- Target network update: every 100 steps

## Bloom Filter Configuration

- Size: 2,000 bits (250 bytes)
- Hash functions: 4 (MurmurHash3)
- False positive rate: ~1%
- Update frequency: After each cache operation

## Baseline Algorithms

- **NoCache**: Lower bound (caching disabled)
- **LRU+LCE**: Least Recently Used with Leave Copy Everywhere
- **LFU+ProbCache**: Least Frequently Used with Probabilistic Caching
- **Combined+LCD**: Combined recency/frequency with Leave Copy Down
- **OPT-Belady**: Optimal offline algorithm (theoretical upper bound)
- **FeiWang-ICC2023**: State-of-the-art multi-agent DQN with exact state exchange
- **FullCache**: Upper bound approximation with large caches

## Documentation

- [DQN Architecture Report](docs/DQN_ARCHITECTURE_REPORT.md) - Detailed system design
- [Research Methodology](docs/RESEARCH_METHODOLOGY.md) - Experimental setup and evaluation

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{tatiparthi2025marl,
  title={Multi-Agent Deep Reinforcement Learning for NDN Caching with 
         Neighbor-Aware State Representation Using Bloom Filters},
  author={Tatiparthi, Sai Praveen},
  school={San Jos{\'e} State University},
  year={2025},
  type={Master's Thesis}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- San José State University, Department of Computer Science
- Thesis Advisor: Dr. Genya Ishigaki


# Simulation and Training Logs

This directory contains logs from various simulation runs and DQN training sessions.

## Benchmark Logs

| File | Description |
|------|-------------|
| `benchmark_run_20251121_031344.log` | Full benchmark run with all algorithms |
| `benchmark_run_20251121_182058.log` | Additional benchmark validation run |
| `benchmark_optimized_20251121_184205.log` | Optimized DQN benchmark run |
| `benchmark_output.log` | Standard benchmark output |

## DQN Training Logs

| File | Description |
|------|-------------|
| `dqn_run.log` | Initial DQN training run |
| `dqn_run_fixed.log` | DQN run with bug fixes |
| `dqn_test_training_phase.log` | Training phase validation |
| `dqn_extended_run_v3_diagnostics.log` | Extended run with diagnostics |
| `dqn_feiwang_run.log` | Comparison with Fei Wang baseline |

## Simulation Logs

| File | Description |
|------|-------------|
| `full_simulation.log` | Complete simulation run output |
| `full_simulation_dqn.log` | Full simulation with DQN enabled |
| `simulation_results.log` | Summary of simulation results |
| `simulation_results_detailed.log` | Detailed simulation output |
| `network_setup.log` | Network initialization log |

## 500-Node Scalability Test

| File | Description |
|------|-------------|
| `500_nodes/run_20251121_194051.log` | Large-scale 500-node simulation |

## Baseline Results

| File | Description |
|------|-------------|
| `Baseline results.txt` | Summary of all baseline algorithm results |

## Log Format

Each log typically contains:
- Timestamp for each operation
- Algorithm configuration details
- Per-request cache hit/miss information
- Round-by-round performance metrics
- Final aggregated statistics


# Model Checkpoints

This directory contains saved model states and checkpoints from DQN training runs.

## Checkpoint Structure

Each checkpoint directory contains:

| File | Description |
|------|-------------|
| `network_state.pkl` | Pickled network state including DQN weights |
| `checkpoint_metrics.txt` | Training metrics at checkpoint time |
| `checkpoint_cache_stats.txt` | Cache statistics summary |
| `checkpoint_final_stats.txt` | Final performance statistics |
| `checkpoint_summary.txt` | Overall training summary |
| `results.log` | Detailed results log |

## Available Checkpoints

### checkpoint_20251117_174955
- **Date**: November 17, 2025
- **Network Size**: 50 nodes
- **Training Rounds**: 40
- **Cache Hit Rate Achieved**: ~31.17%

## Loading Checkpoints

```python
import pickle

with open('checkpoints/checkpoint_20251117_174955/network_state.pkl', 'rb') as f:
    state = pickle.load(f)
    
# Access DQN agent states
for router_id, router_state in state['routers'].items():
    if 'dqn_weights' in router_state:
        weights = router_state['dqn_weights']
```

## Training New Models

To train and save new checkpoints:

```bash
python run_experiment.py --scenario scenario_medium.json --save-checkpoint
```


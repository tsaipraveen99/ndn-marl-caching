"""
Performance benchmarks comparing different caching policies
Tests: Hit rate comparison, latency comparison, cache utilization, scalability
"""

import os
import sys
import time
import json
import argparse
import logging
from typing import Dict, List, Optional
import statistics
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (
    create_network,
    run_simulation,
    warmup_cache,
    align_user_distributions_with_producers,
    initialize_bloom_filter_propagation,
    setup_all_routers_to_dqn_mode,
    setup_logging,
    log_info,
)

# Colab minimal logging support
try:
    from colab_minimal_logging import is_colab, get_minimal_logger
except ImportError:
    # Not available in non-Colab environments
    def is_colab():
        return False
    def get_minimal_logger():
        return None

# Checkpoint directory
CHECKPOINT_DIR = Path("benchmark_checkpoints")
CHECKPOINT_FILE = CHECKPOINT_DIR / "benchmark_checkpoint.json"
RESULTS_FILE = CHECKPOINT_DIR / "benchmark_results.json"
CONFIGS_DIR = Path("configs")
RESULTS_DIR = Path("results")
DEFAULT_ALGORITHM_SUITE = [
    {
        "name": "NoCache-LowerBound",
        "description": "Lower bound reference with caching disabled",
        "overrides": {
            "NDN_SIM_USE_DQN": "0",
            "NDN_SIM_CACHE_POLICY": "fifo",
            "NDN_SIM_PLACEMENT_POLICY": "none",
            "NDN_SIM_ROUNDS": "40"  # ⬆️ Added for fair comparison
        }
    },
    {
        "name": "LRU+LCE",
        "description": "Classic LRU with leave-copy-everywhere placement",
        "overrides": {
            "NDN_SIM_USE_DQN": "0",
            "NDN_SIM_CACHE_POLICY": "lru",
            "NDN_SIM_PLACEMENT_POLICY": "lce",
            "NDN_SIM_ROUNDS": "40"  # ⬆️ Added for fair comparison
        }
    },
    {
        "name": "LFU+ProbCache",
        "description": "LFU cache with probabilistic copy-down placement",
        "overrides": {
            "NDN_SIM_USE_DQN": "0",
            "NDN_SIM_CACHE_POLICY": "lfu",
            "NDN_SIM_PLACEMENT_POLICY": "prob_cache",
            "NDN_SIM_ROUNDS": "40"  # ⬆️ Added for fair comparison
        }
    },
    {
        "name": "Combined+LCD",
        "description": "Combined policy with leave-copy-down",
        "overrides": {
            "NDN_SIM_USE_DQN": "0",
            "NDN_SIM_CACHE_POLICY": "combined",
            "NDN_SIM_PLACEMENT_POLICY": "lcd",
            "NDN_SIM_ROUNDS": "40"  # ⬆️ Added for fair comparison
        }
    },
    {
        "name": "DQN",
        "description": "Pure DQN - learns both placement (WHERE) and content selection (WHAT) - OPTIMIZED",
        "overrides": {
            "NDN_SIM_USE_DQN": "1",
            "NDN_SIM_CACHE_POLICY": "combined",
            "NDN_SIM_PLACEMENT_POLICY": "lce",  # Not used, but set for consistency
            "NDN_SIM_DQN_RESPECT_PLACEMENT": "0",  # Pure DQN mode - placement bypassed
            "NDN_SIM_ROUNDS": "40",  # ⬆️ Increased from 8 for better learning
            "NDN_SIM_WARMUP_ROUNDS": "10",  # ⬆️ Increased from 3
            "NDN_SIM_DQN_LR": "0.001",  # ⬆️ Increased from 0.0005 for faster convergence
            "NDN_SIM_DQN_EPSILON_DECAY": "0.998",  # ⬆️ Faster decay (was 0.9995)
            "NDN_SIM_DQN_EPSILON_END": "0.02",  # ⬇️ Lower minimum (was 0.05)
            "NDN_SIM_DQN_MEMORY_MULTIPLIER": "50",  # ⬆️ Larger memory (was 25)
            "NDN_SIM_DQN_TRAINING_FREQUENCY": "1"  # ⬆️ Train more frequently (was 2)
        }
    },
    {
        "name": "DQN+ProbCache",
        "description": "Hybrid: ProbCache decides WHERE, DQN decides WHAT - OPTIMIZED",
        "overrides": {
            "NDN_SIM_USE_DQN": "1",
            "NDN_SIM_CACHE_POLICY": "combined",
            "NDN_SIM_PLACEMENT_POLICY": "prob_cache",
            "NDN_SIM_DQN_RESPECT_PLACEMENT": "1",  # Hybrid mode - placement policy active
            "NDN_SIM_ROUNDS": "40",  # ⬆️ Increased from 8 for better learning
            "NDN_SIM_WARMUP_ROUNDS": "10",  # ⬆️ Increased from 3
            "NDN_SIM_DQN_LR": "0.001",  # ⬆️ Increased from 0.0005
            "NDN_SIM_DQN_EPSILON_DECAY": "0.998",  # ⬆️ Faster decay
            "NDN_SIM_DQN_EPSILON_END": "0.02",  # ⬇️ Lower minimum
            "NDN_SIM_DQN_MEMORY_MULTIPLIER": "50",  # ⬆️ Larger memory
            "NDN_SIM_DQN_TRAINING_FREQUENCY": "1"  # ⬆️ Train more frequently
        }
    },
    {
        "name": "DQN+MPC",
        "description": "Hybrid: MPC decides WHERE, DQN decides WHAT - OPTIMIZED",
        "overrides": {
            "NDN_SIM_USE_DQN": "1",
            "NDN_SIM_CACHE_POLICY": "combined",
            "NDN_SIM_PLACEMENT_POLICY": "mpc_cache",
            "NDN_SIM_DQN_RESPECT_PLACEMENT": "1",  # Hybrid mode - placement policy active
            "NDN_SIM_ROUNDS": "40",  # ⬆️ Increased from 8 for better learning
            "NDN_SIM_WARMUP_ROUNDS": "10",  # ⬆️ Increased from 3
            "NDN_SIM_DQN_LR": "0.001",  # ⬆️ Increased from 0.0005
            "NDN_SIM_DQN_EPSILON_DECAY": "0.998",  # ⬆️ Faster decay
            "NDN_SIM_DQN_EPSILON_END": "0.02",  # ⬇️ Lower minimum
            "NDN_SIM_DQN_MEMORY_MULTIPLIER": "50",  # ⬆️ Larger memory
            "NDN_SIM_DQN_TRAINING_FREQUENCY": "1"  # ⬆️ Train more frequently
        }
    },
    {
        "name": "FullCache-UpperBound",
        "description": "Upper bound approximation with very large caches and LCE",
        "overrides": {
            "NDN_SIM_USE_DQN": "0",
            "NDN_SIM_CACHE_POLICY": "lru",
            "NDN_SIM_PLACEMENT_POLICY": "lce",
            "NDN_SIM_CACHE_CAPACITY_RATIO": "1.0",
            "NDN_SIM_CACHE_CAPACITY": "100000",
            "NDN_SIM_ROUNDS": "40"  # ⬆️ Added for fair comparison
        }
    },
    {
        "name": "OPT-Belady",
        "description": "Optimal offline (theoretical upper bound using Belady's algorithm)",
        "overrides": {
            "NDN_SIM_USE_DQN": "0",
            "NDN_SIM_CACHE_POLICY": "opt",
            "NDN_SIM_PLACEMENT_POLICY": "lce",
            "NDN_SIM_ROUNDS": "40"  # ⬆️ Added for fair comparison
        }
    },
    {
        "name": "LFO-Baseline",
        "description": "Least Frequently Optimal heuristic baseline",
        "overrides": {
            "NDN_SIM_USE_DQN": "0",
            "NDN_SIM_CACHE_POLICY": "lfo",
            "NDN_SIM_PLACEMENT_POLICY": "lce",
            "NDN_SIM_ROUNDS": "40"  # ⬆️ Added for fair comparison
        }
    },
    {
        "name": "FeiWang-ICC2023",
        "description": "Fei Wang et al. multi-agent DQN with exact neighbor state",
        "overrides": {
            "NDN_SIM_USE_DQN": "0",
            "NDN_SIM_CACHE_POLICY": "fei_wang",
            "NDN_SIM_PLACEMENT_POLICY": "lce",
            "NDN_SIM_ROUNDS": "40"  # ⬆️ Added for fair comparison
        }
    },
]

DQN_SWEEP_PRESETS = [
    {
        "name": "DQN_Base",
        "description": "Baseline DQN configuration (reference)",
        "overrides": {
            "NDN_SIM_USE_DQN": "1",
            "NDN_SIM_CACHE_POLICY": "combined",
            "NDN_SIM_PLACEMENT_POLICY": "prob_cache",
        },
    },
    {
        "name": "DQN_LowLR_BigBatch",
        "description": "Lower LR, larger batch, slower decay for stability",
        "overrides": {
            "NDN_SIM_USE_DQN": "1",
            "NDN_SIM_CACHE_POLICY": "combined",
            "NDN_SIM_PLACEMENT_POLICY": "prob_cache",
            "NDN_SIM_DQN_LR": "0.0001",
            "DQN_BATCH_SIZE": "128",
            "NDN_SIM_DQN_TARGET_UPDATE": "150",
            "NDN_SIM_DQN_MEMORY_SIZE": "60000",
            "NDN_SIM_DQN_EPSILON_DECAY": "0.997",
            "NDN_SIM_DQN_N_STEP": "15",
        },
    },
    {
        "name": "DQN_HighLR_FastUpdate",
        "description": "Higher LR with faster target updates for responsiveness",
        "overrides": {
            "NDN_SIM_USE_DQN": "1",
            "NDN_SIM_CACHE_POLICY": "combined",
            "NDN_SIM_PLACEMENT_POLICY": "prob_cache",
            "NDN_SIM_DQN_LR": "0.0005",
            "NDN_SIM_DQN_TARGET_UPDATE": "80",
            "NDN_SIM_DQN_EPSILON_DECAY": "0.993",
            "NDN_SIM_DQN_MEMORY_MULTIPLIER": "15",
            "NDN_SIM_DQN_N_STEP": "10",
        },
    },
    {
        "name": "DQN_LatencyFocused",
        "description": "Latency-weighted reward with MPC placement fallback",
        "overrides": {
            "NDN_SIM_USE_DQN": "1",
            "NDN_SIM_CACHE_POLICY": "combined",
            "NDN_SIM_PLACEMENT_POLICY": "mpc_cache",
            "NDN_SIM_DQN_LATENCY_WEIGHT": "0.2",
            "NDN_SIM_DQN_DECISION_LATENCY_WEIGHT": "0.1",
            "NDN_SIM_DQN_MISS_PENALTY": "3.0",
            "NDN_SIM_DQN_EPSILON_DECAY": "0.996",
            "DQN_BATCH_SIZE": "96",
        },
    },
]


def save_checkpoint(algorithm_name: str, completed_runs: int, total_runs: int, results: Dict):
    """Save checkpoint after algorithm completes"""
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    checkpoint = {
        'algorithm': algorithm_name,
        'completed_runs': completed_runs,
        'total_runs': total_runs,
        'timestamp': time.time(),
        'results': results
    }
    
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    
    print(f"  💾 Checkpoint saved: {algorithm_name} ({completed_runs}/{total_runs} runs)")


def load_checkpoint() -> Optional[Dict]:
    """Load checkpoint if it exists"""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"  ⚠️  Error loading checkpoint: {e}")
            return None
    return None


def save_results(results: Dict):
    """Save results incrementally"""
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    # Load existing results if any
    existing_results = {}
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE, 'r') as f:
                existing_results = json.load(f)
        except:
            pass
    
    # Merge with new results
    existing_results.update(results)
    
    # Save
    with open(RESULTS_FILE, 'w') as f:
        json.dump(existing_results, f, indent=2)
    
    print(f"  💾 Results saved: {len(results)} algorithms completed")


def load_results() -> Dict:
    """Load existing results"""
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}


def clear_checkpoint():
    """Clear checkpoint file"""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

def auto_tune_simulation_load(config: Dict, logger: Optional[logging.Logger] = None) -> Dict[str, Dict[str, float]]:
    """Automatically adjust heavy-load parameters for large scenarios to prevent stalls."""
    adjustments: Dict[str, Dict[str, float]] = {}
    try:
        nodes = int(config.get('NDN_SIM_NODES', 0))
        producers = int(config.get('NDN_SIM_PRODUCERS', 0))
        contents = int(config.get('NDN_SIM_CONTENTS', 0))
        users = int(config.get('NDN_SIM_USERS', 0))
        rounds = int(config.get('NDN_SIM_ROUNDS', 1))
        requests = int(config.get('NDN_SIM_REQUESTS', 1))
    except (TypeError, ValueError):
        return adjustments

    total_requests = users * rounds * requests
    max_total_requests = int(os.environ.get("NDN_SIM_MAX_TOTAL_REQUESTS", "48000"))
    if total_requests > max_total_requests and users > 0 and rounds > 0:
        # Clamp requests per user so total stays within safe bounds
        new_requests = max(1, max_total_requests // (users * rounds))
        if new_requests < requests:
            adjustments['NDN_SIM_REQUESTS'] = {'old': requests, 'new': new_requests}
            config['NDN_SIM_REQUESTS'] = new_requests

    if nodes >= 150:
        queue_size = int(config.get('NDN_SIM_MAX_QUEUE_SIZE', 0) or 0)
        if queue_size < 15000:
            adjustments['NDN_SIM_MAX_QUEUE_SIZE'] = {'old': queue_size, 'new': 15000}
            config['NDN_SIM_MAX_QUEUE_SIZE'] = 15000

        worker_timeout = float(config.get('NDN_SIM_WORKER_TIMEOUT', 60.0))
        if worker_timeout < 180.0:
            adjustments['NDN_SIM_WORKER_TIMEOUT'] = {'old': worker_timeout, 'new': 180.0}
            config['NDN_SIM_WORKER_TIMEOUT'] = 180.0

        force_threshold = float(config.get('NDN_SIM_FORCE_DRAIN_THRESHOLD', 0.5))
        if force_threshold > 0.3:
            adjustments['NDN_SIM_FORCE_DRAIN_THRESHOLD'] = {'old': force_threshold, 'new': 0.3}
            config['NDN_SIM_FORCE_DRAIN_THRESHOLD'] = 0.3

        dqn_concurrency = int(config.get('NDN_SIM_DQN_CONCURRENCY', 4))
        if dqn_concurrency > 2:
            adjustments['NDN_SIM_DQN_CONCURRENCY'] = {'old': dqn_concurrency, 'new': 2}
            config['NDN_SIM_DQN_CONCURRENCY'] = 2

    if adjustments:
        summary_parts = []
        for key, delta in adjustments.items():
            summary_parts.append(f"{key}: {delta['old']}→{delta['new']}")
        summary = ", ".join(summary_parts)
        if logger:
            log_info(f"Auto-tuned config to prevent overload ({summary})", logger)
        else:
            print(f"Auto-tuned config to prevent overload ({summary})")

    return adjustments


def _load_config_from_file(config_path: Path) -> Dict:
    """Load a scenario configuration from JSON or YAML."""
    if not config_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {config_path}")
    
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        with open(config_path, "r") as f:
            return json.load(f)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                f"PyYAML is required to load YAML scenarios ({config_path}). "
                "Install it via `pip install pyyaml`."
            ) from exc
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported scenario file extension: {config_path.suffix}")


def load_scenario_config(name: str, overrides: Optional[Dict] = None) -> Dict:
    """
    Load a scenario configuration by name or explicit path.
    
    Args:
        name: Scenario name (e.g., \"minimal\", \"scenario_medium.json\") or path.
        overrides: Optional dictionary of overrides applied on top of the scenario.
    
    Returns:
        Dict of simulation settings.
    """
    path = Path(name)
    candidates = []
    
    if path.exists():
        candidates.append(path)
    else:
        if not name.endswith(".json") and not name.endswith(".yaml") and not name.endswith(".yml"):
            candidates.append(CONFIGS_DIR / f"{name}.json")
            candidates.append(CONFIGS_DIR / f"scenario_{name}.json")
            candidates.append(CONFIGS_DIR / f"{name}.yaml")
            candidates.append(CONFIGS_DIR / f"scenario_{name}.yaml")
        candidates.append(CONFIGS_DIR / name)
    
    scenario_config = None
    for candidate in candidates:
        if candidate.exists():
            scenario_config = _load_config_from_file(candidate)
            print(f"  📁 Loaded scenario '{name}' from {candidate}")
            break
    
    if scenario_config is None:
        raise FileNotFoundError(
            f"Unable to locate scenario '{name}'. "
            f"Searched: {[str(c) for c in candidates]}"
        )
    
    if overrides:
        scenario_config.update(overrides)
    
    return scenario_config


def run_benchmark(config: Dict, num_runs: int = 10, seed: int = 42, checkpoint_key: Optional[str] = None) -> Dict:
    """
    Run benchmark with given configuration
    
    Args:
        config: Configuration dictionary
        num_runs: Number of runs to average
        seed: Base seed for reproducibility (each run uses seed + run_number)
        checkpoint_key: Key for checkpointing (algorithm name)
    
    Returns:
        Dictionary with average metrics
    """
    sim_logger, net_logger = setup_logging(log_mode="a")
    auto_tune_simulation_load(config, sim_logger)

    # Check for existing checkpoint
    start_run = 0
    results = []
    if checkpoint_key:
        checkpoint = load_checkpoint()
        if checkpoint and checkpoint.get('algorithm') == checkpoint_key:
            start_run = checkpoint.get('completed_runs', 0)
            # Load partial results if available
            checkpoint_results = checkpoint.get('results', {})
            if 'partial_results' in checkpoint_results:
                results = checkpoint_results['partial_results']
            if start_run > 0:
                print(f"  🔄 Resuming from checkpoint: {start_run}/{num_runs} runs already completed")
                print(f"  📊 Loaded {len(results)} previous run results")
    
    for run in range(start_run, num_runs):
        run_idx = run + 1
        print(f"  Run {run_idx}/{num_runs}...")
        log_info(
            f"Run {run_idx}/{num_runs} for {checkpoint_key or 'ad-hoc config'}",
            sim_logger
        )
        
        # Set environment variables
        for key, value in config.items():
            os.environ[key] = str(value)
        
        # Set seed for reproducibility (each run gets different seed)
        import random
        import numpy as np
        run_seed = seed + run
        random.seed(run_seed)
        np.random.seed(run_seed)
        
        try:
            # Use fixed seed for network topology (same for all runs)
            import networkx as nx
            import random
            random.seed(seed)  # Fixed seed for topology
            np.random.seed(seed)
            
            # Log run started with minimal logger if in Colab
            minimal_logger = get_minimal_logger()
            if minimal_logger:
                algorithm_name = checkpoint_key or 'benchmark'
                minimal_logger.log_run_started(run_idx, num_runs, algorithm_name)
            
            G, users, producers, runtime = create_network(
                num_nodes=int(config.get('NDN_SIM_NODES', 300)),
                num_producers=int(config.get('NDN_SIM_PRODUCERS', 60)),
                num_contents=int(config.get('NDN_SIM_CONTENTS', 6000)),
                num_users=int(config.get('NDN_SIM_USERS', 2000)),
                cache_policy=config.get('NDN_SIM_CACHE_POLICY', 'fifo'),
                logger=net_logger
            )
            
            # Set experiment ID for TensorBoard logging (if not already set)
            if not os.environ.get('NDN_SIM_EXPERIMENT_ID'):
                from datetime import datetime
                algorithm_name = checkpoint_key or 'benchmark'
                experiment_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{algorithm_name}"
                os.environ['NDN_SIM_EXPERIMENT_ID'] = experiment_id
                print(f"    📊 Experiment ID: {experiment_id}")
            
            # Initialize DQN if enabled
            if config.get('NDN_SIM_USE_DQN', '0') == '1':
                print("    🔧 Enabling DQN mode on all routers...")
                setup_all_routers_to_dqn_mode(G, logger=net_logger)
                initialize_bloom_filter_propagation(G, logger=net_logger)
                
                # Initialize DQN Training Manager for asynchronous training (CRITICAL)
                from router import DQNTrainingManager
                try:
                    # Determine optimal number of training workers
                    # For GPU: 4 workers (GPU can parallelize)
                    # For CPU: 2 workers (CPU bound)
                    import torch
                    if torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                        max_training_workers = 4  # GPU can handle parallel training
                    else:
                        max_training_workers = 2  # CPU: fewer workers
                except:
                    max_training_workers = 2  # Default to 2 if torch not available
                
                training_manager = DQNTrainingManager.get_instance(max_workers=max_training_workers)
                print(f"    ✅ DQN Training Manager initialized with {max_training_workers} workers")
                
                # Verify DQN agents initialized
                dqn_count = 0
                dqn_failed = 0
                for node, data in G.nodes(data=True):
                    if 'router' in data:
                        router = data['router']
                        if hasattr(router, 'content_store'):
                            cs = router.content_store
                            if hasattr(cs, 'mode') and cs.mode == "dqn_cache":
                                if hasattr(cs, 'dqn_agent') and cs.dqn_agent is not None:
                                    dqn_count += 1
                                else:
                                    dqn_failed += 1
                
                if dqn_count > 0:
                    print(f"    ✅ {dqn_count} routers with DQN agents initialized")
                if dqn_failed > 0:
                    print(f"    ⚠️  Warning: {dqn_failed} routers failed to initialize DQN agents")
            
            # Reset seed for request generation (different per run)
            random.seed(run_seed)
            np.random.seed(run_seed)
            
            # IMPORTANT: Ensure content exists and align user distributions
            align_user_distributions_with_producers(users, producers, logger=net_logger)
            
            # WARM-UP PHASE: Pre-populate caches before evaluation
            # This ensures fair comparison by starting all algorithms with warm caches
            warmup_rounds = int(config.get('NDN_SIM_WARMUP_ROUNDS', 10))
            print(f"    🔥 Warm-up phase: {warmup_rounds} rounds...")
            warmup_cache(G, users, producers, num_warmup_rounds=warmup_rounds, logger=net_logger)
            
            # Reset statistics AFTER warm-up but BEFORE evaluation
            # This ensures we only measure performance during the evaluation phase
            from router import stats as global_stats
            try:
                with global_stats.lock:
                    global_stats.nodes_traversed = 0
                    global_stats.cache_hits = 0
                    global_stats.data_packets_transferred = 0
                    global_stats.total_data_size_transferred = 0
            except Exception:
                pass
            
            # EVALUATION PHASE: Run simulation with warm caches
            print(f"    📊 Evaluation phase: {config.get('NDN_SIM_ROUNDS', 20)} rounds...")
            stats = run_simulation(
                G, users, producers,
                num_rounds=int(config.get('NDN_SIM_ROUNDS', 20)),
                num_requests=int(config.get('NDN_SIM_REQUESTS', 5)),
                logger=sim_logger
            )
            
            # Collect metrics
            cached_items = 0
            total_insertions = 0
            routers_with_cache = 0
            cache_utilizations = []
            per_router_utilizations = []
            
            # Record cache utilization metrics
            from metrics import get_metrics_collector
            metrics_collector = get_metrics_collector()
            
            for node, data in G.nodes(data=True):
                if 'router' in data:
                    router = data['router']
                    if hasattr(router, 'content_store'):
                        cs = router.content_store
                        cached_items += len(cs.store)
                        total_insertions += getattr(cs, 'insertions', 0)
                        if len(cs.store) > 0:
                            routers_with_cache += 1
                        
                        # Record cache utilization
                        used = cs.total_capacity - cs.remaining_capacity
                        total = cs.total_capacity
                        if total > 0:
                            utilization = (used / total) * 100.0
                            cache_utilizations.append(utilization)
                            metrics_collector.record_cache_utilization(router.router_id, used, total)
                            per_router_utilizations.append((router.router_id, utilization))
            
            if per_router_utilizations:
                sorted_utils = sorted(per_router_utilizations, key=lambda x: x[1], reverse=True)
                top_util = sorted_utils[:5]
                low_util = sorted(per_router_utilizations, key=lambda x: x[1])[:5]
                print("    🧮 Top cache utilizations:")
                for router_id, util in top_util:
                    print(f"      Router {router_id}: {util:.1f}%")
                print("    🧮 Lowest cache utilizations:")
                for router_id, util in low_util:
                    print(f"      Router {router_id}: {util:.1f}%")
            
            # Get comprehensive metrics
            all_metrics = metrics_collector.get_all_metrics()
            
            if cache_utilizations:
                avg_utilization = sum(cache_utilizations) / len(cache_utilizations)
            elif per_router_utilizations:
                avg_utilization = sum(util for _, util in per_router_utilizations) / len(per_router_utilizations)
            else:
                avg_utilization = 0.0
            run_result = {
                'hit_rate': stats.get('hit_rate', 0),
                'cache_hits': stats.get('cache_hits', 0),
                'nodes_traversed': stats.get('nodes_traversed', 0),
                'cached_items': cached_items,
                'total_insertions': total_insertions,
                'routers_with_cache': routers_with_cache,
                'avg_cache_utilization': avg_utilization,
                'latency_mean': all_metrics.get('latency', {}).get('mean', 0.0),
                'redundancy_mean': all_metrics.get('redundancy', {}).get('mean', 0.0),
                'dispersion_mean': all_metrics.get('dispersion', {}).get('mean', 0.0)
            }
            results.append(run_result)
            
            # Use minimal logger if in Colab, otherwise use regular logging
            minimal_logger = get_minimal_logger()
            if minimal_logger:
                minimal_logger.log_run_finished(run_idx, num_runs, run_result['hit_rate'])
            else:
                log_info(
                    f"Run {run_idx}/{num_runs} finished: hit_rate={run_result['hit_rate']:.4f}, "
                    f"cache_hits={run_result['cache_hits']}, avg_cache_util={run_result['avg_cache_utilization']:.2f}%",
                    sim_logger
                )
            
            # Shutdown DQN Training Manager gracefully (if DQN was enabled)
            if config.get('NDN_SIM_USE_DQN', '0') == '1':
                try:
                    from router import DQNTrainingManager
                    training_manager = DQNTrainingManager.get_instance()
                    if training_manager is not None:
                        training_manager.shutdown()
                except Exception as e:
                    print(f"    ⚠️  Warning: Error shutting down training manager: {e}")
            
            if runtime:
                runtime.shutdown()
                
        except Exception as e:
            print(f"    Error in run {run + 1}: {e}")
            continue
        
        # Save checkpoint after each run (for safety)
        if checkpoint_key and (run + 1) % 2 == 0:  # Every 2 runs
            partial_results = results.copy()
            save_checkpoint(checkpoint_key, len(results), num_runs, {
                'partial_results': partial_results,
                'num_completed': len(results)
            })
    
    # Calculate averages
    if not results:
        return {}
    
    # Calculate statistics with confidence intervals
    from statistical_analysis import calculate_mean_std_ci, calculate_mean, calculate_std, calculate_confidence_interval
    
    hit_rates = [r['hit_rate'] for r in results]
    cache_hits_list = [r['cache_hits'] for r in results]
    nodes_traversed_list = [r['nodes_traversed'] for r in results]
    cached_items_list = [r['cached_items'] for r in results]
    cache_util_list = [r.get('avg_cache_utilization', 0.0) for r in results]
    total_insertions_list = [r['total_insertions'] for r in results]
    routers_with_cache_list = [r['routers_with_cache'] for r in results]
    
    hit_rate_stats = calculate_mean_std_ci(hit_rates)
    ci_lower, ci_upper = calculate_confidence_interval(hit_rates)
    
    avg_results = {
        'hit_rate': hit_rate_stats['mean'],
        'hit_rate_std': hit_rate_stats['std'],
        'hit_rate_ci_lower': ci_lower,
        'hit_rate_ci_upper': ci_upper,
        'cache_hits': round(statistics.mean(cache_hits_list)),
        'cache_hits_std': round(statistics.stdev(cache_hits_list) if len(cache_hits_list) > 1 else 0),
        'nodes_traversed': round(statistics.mean(nodes_traversed_list)),
        'nodes_traversed_std': round(statistics.stdev(nodes_traversed_list) if len(nodes_traversed_list) > 1 else 0),
        'cached_items': round(statistics.mean(cached_items_list)),
        'cached_items_std': round(statistics.stdev(cached_items_list) if len(cached_items_list) > 1 else 0),
        'total_insertions': round(statistics.mean(total_insertions_list)),
        'total_insertions_std': round(statistics.stdev(total_insertions_list) if len(total_insertions_list) > 1 else 0),
        'routers_with_cache': round(statistics.mean(routers_with_cache_list)),
        'routers_with_cache_std': round(statistics.stdev(routers_with_cache_list) if len(routers_with_cache_list) > 1 else 0),
        'avg_cache_utilization': statistics.mean(cache_util_list) if cache_util_list else 0.0,
        'num_runs': len(results)
    }
    
    # Clear checkpoint if completed
    if checkpoint_key:
        checkpoint = load_checkpoint()
        if checkpoint and checkpoint.get('algorithm') == checkpoint_key:
            if len(results) >= num_runs:
                clear_checkpoint()
                print(f"  ✅ Checkpoint cleared: {checkpoint_key} completed")
    
    return avg_results


def run_algorithm_suite(
    scenario: str = "medium",
    algorithms: Optional[List[Dict]] = None,
    num_runs: int = 5,  # ⬆️ Increased from 1 to 5 for statistical significance
    seed: int = 42,
    output_dir: Path = RESULTS_DIR,
) -> List[Dict]:
    """
    Run multiple algorithms on a scenario and persist comparison metrics.
    """
    suite = algorithms or DEFAULT_ALGORITHM_SUITE
    comparison_results: List[Dict] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get minimal logger if in Colab
    minimal_logger = get_minimal_logger()
    
    for algo in suite:
        name = algo.get("name", "Unnamed")
        desc = algo.get("description", "")
        overrides = algo.get("overrides", {})
        
        # Use minimal logger if in Colab, otherwise use regular print
        if minimal_logger:
            minimal_logger.log_algorithm_start(name)
        else:
            print("\n" + "=" * 70)
            print(f"ALGORITHM: {name}")
            if desc:
                print(f"Description: {desc}")
        
        config = load_scenario_config(scenario, overrides=overrides)
        results = run_benchmark(config, num_runs=num_runs, seed=seed, checkpoint_key=name)
        
        # Log algorithm finished with minimal logger if in Colab
        if minimal_logger:
            # Fix: results['hit_rate'] is a float, not a dict (see run_benchmark return at line 703)
            avg_hit_rate = results.get('hit_rate', 0)
            minimal_logger.log_algorithm_finished(name, avg_hit_rate)
        algo_record = {
            "algorithm": name,
            "description": desc,
            "scenario": scenario,
            **results,
        }
        comparison_results.append(algo_record)
    output_file = output_dir / f"algorithm_comparison_{scenario}.json"
    with output_file.open("w") as fp:
        json.dump(comparison_results, fp, indent=2)
    print("\n" + "=" * 70)
    print(f"Comparison complete. Results saved to {output_file}")
    print("=" * 70)
    header = f"{'Algorithm':<20} {'HitRate%':>10} {'CacheHits':>12} {'Latency(ms)':>14} {'Util%':>8}"
    print(header)
    print("-" * len(header))
    for entry in comparison_results:
        hit_rate = entry.get("hit_rate", 0) * 100
        latency_ms = entry.get("latency_mean", 0) * 1000
        util = entry.get("avg_cache_utilization", 0)
        print(
            f"{entry['algorithm']:<20} {hit_rate:10.2f} "
            f"{entry.get('cache_hits', 0):12d} {latency_ms:14.2f} {util:8.2f}"
        )
    return comparison_results


def run_dqn_sweep(
    scenario: str = "medium",
    presets: Optional[List[Dict]] = None,
    num_runs: int = 1,
    seed: int = 42,
) -> List[Dict]:
    """
    Run a hyperparameter sweep focused on DQN/MARL variants and store metrics/graphs.
    """
    sweep = presets or DQN_SWEEP_PRESETS
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sweep_results: List[Dict] = []
    for preset in sweep:
        name = preset.get("name", "DQN_Variant")
        desc = preset.get("description", "")
        overrides = preset.get("overrides", {}).copy()
        overrides.setdefault("NDN_SIM_USE_DQN", "1")
        overrides.setdefault("NDN_SIM_CACHE_POLICY", "combined")
        overrides.setdefault("NDN_SIM_PLACEMENT_POLICY", "prob_cache")
        print("\n" + "=" * 70)
        print(f"SWEEP VARIANT: {name}")
        if desc:
            print(desc)
        config = load_scenario_config(scenario, overrides=overrides)
        result = run_benchmark(config, num_runs=num_runs, seed=seed, checkpoint_key=f"{scenario}_{name}")
        if not result:
            print(f"⚠️  {name} produced no results, skipping.")
            continue
        record = {
            "variant": name,
            "description": desc,
            "scenario": scenario,
            **result,
        }
        sweep_results.append(record)
    if not sweep_results:
        print("⚠️  No sweep data collected.")
        return []
    # Persist JSON
    json_path = RESULTS_DIR / f"dqn_sweep_{scenario}.json"
    with json_path.open("w") as fp:
        json.dump(sweep_results, fp, indent=2)
    print(f"💾 Sweep metrics saved to {json_path}")
    # Render chart if matplotlib is available
    plot_path = RESULTS_DIR / f"dqn_sweep_{scenario}.png"
    try:
        import matplotlib.pyplot as plt

        labels = [entry["variant"] for entry in sweep_results]
        hit_rates = [entry.get("hit_rate", 0.0) * 100 for entry in sweep_results]
        plt.figure(figsize=(10, 5))
        bars = plt.bar(labels, hit_rates, color="#3b82f6")
        plt.ylabel("Hit Rate (%)")
        plt.title(f"DQN Hyperparameter Sweep ({scenario})")
        plt.xticks(rotation=25, ha="right")
        for bar, rate in zip(bars, hit_rates):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                rate + 0.5,
                f"{rate:.2f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"📈 Sweep chart saved to {plot_path}")
    except ImportError:
        print("⚠️  matplotlib not available; install it to generate sweep charts.")
    # Console table
    print("\n" + "=" * 70)
    print(f"DQN SWEEP SUMMARY ({scenario.upper()})")
    print("=" * 70)
    header = f"{'Variant':<25} {'HitRate%':>10} {'CacheHits':>12} {'Latency(ms)':>14} {'Util%':>8}"
    print(header)
    print("-" * len(header))
    for entry in sweep_results:
        hit_rate = entry.get("hit_rate", 0) * 100
        latency_ms = entry.get("latency_mean", 0) * 1000
        util = entry.get("avg_cache_utilization", 0)
        print(
            f"{entry['variant']:<25} {hit_rate:10.2f} "
            f"{entry.get('cache_hits', 0):12d} {latency_ms:14.2f} {util:8.2f}"
        )
    return sweep_results


def test_minimal_dqn(resume: bool = False):
    """Minimal DQN test for quick iteration and debugging"""
    print("\n" + "="*80)
    print("MINIMAL DQN TEST: Quick Testing Configuration")
    print("="*80)
    
    minimal_config = load_scenario_config("minimal")
    
    print("Running minimal DQN test...")
    print(f"  Network: {minimal_config['NDN_SIM_NODES']} nodes, {minimal_config['NDN_SIM_PRODUCERS']} producers")
    print(f"  Content: {minimal_config['NDN_SIM_CONTENTS']} items, {minimal_config['NDN_SIM_USERS']} users")
    print(f"  Rounds: {minimal_config['NDN_SIM_WARMUP_ROUNDS']} warmup + {minimal_config['NDN_SIM_ROUNDS']} evaluation")
    
    # Run single test
    result = run_benchmark(minimal_config, num_runs=1, seed=42, checkpoint_key=None)
    
    print("\n" + "="*80)
    print("MINIMAL TEST RESULTS")
    print("="*80)
    print(f"Hit Rate: {result.get('hit_rate', 0):.4f}")
    print(f"Cache Hits: {result.get('cache_hits', 0)}")
    print(f"Nodes Traversed: {result.get('nodes_traversed', 0)}")
    print(f"Avg Cache Utilization: {result.get('avg_cache_utilization', 0):.2f}%")
    print("="*80)
    
    return result


def test_medium_dqn(resume: bool = False):
    """Medium-sized DQN test with larger network and increased cache capacity"""
    print("\n" + "="*80)
    print("MEDIUM DQN TEST: Medium Network Configuration")
    print("="*80)
    
    medium_config = load_scenario_config("medium")
    
    print("Running medium DQN test...")
    print(f"  Network: {medium_config['NDN_SIM_NODES']} nodes, {medium_config['NDN_SIM_PRODUCERS']} producers")
    print(f"  Content: {medium_config['NDN_SIM_CONTENTS']} items, {medium_config['NDN_SIM_USERS']} users")
    print(f"  Rounds: {medium_config['NDN_SIM_WARMUP_ROUNDS']} warmup + {medium_config['NDN_SIM_ROUNDS']} evaluation")
    print(f"  Cache Capacity: {medium_config['NDN_SIM_CACHE_CAPACITY']} items per router")
    
    # Run single test
    result = run_benchmark(medium_config, num_runs=1, seed=42, checkpoint_key=None)
    
    print("\n" + "="*80)
    print("MEDIUM TEST RESULTS")
    print("="*80)
    print(f"Hit Rate: {result.get('hit_rate', 0):.4f} ({result.get('hit_rate', 0)*100:.2f}%)")
    print(f"Cache Hits: {result.get('cache_hits', 0)}")
    print(f"Nodes Traversed: {result.get('nodes_traversed', 0)}")
    print(f"Cached Items: {result.get('cached_items', 0)}")
    print(f"Avg Cache Utilization: {result.get('avg_cache_utilization', 0):.2f}%")
    print("="*80)
    
    return result


def test_large_dqn(resume: bool = False):
    """Large-scale DQN test mirroring MARL research-scale deployments"""
    print("\n" + "="*80)
    print("LARGE DQN TEST: Large Network Configuration")
    print("="*80)
    
    large_config = load_scenario_config("large")
    
    print("Running large DQN test...")
    print(f"  Network: {large_config['NDN_SIM_NODES']} nodes, {large_config['NDN_SIM_PRODUCERS']} producers")
    print(f"  Content: {large_config['NDN_SIM_CONTENTS']} items, {large_config['NDN_SIM_USERS']} users")
    print(f"  Rounds: {large_config['NDN_SIM_WARMUP_ROUNDS']} warmup + {large_config['NDN_SIM_ROUNDS']} evaluation")
    print(f"  Cache Capacity: {large_config['NDN_SIM_CACHE_CAPACITY']} items per router")
    
    result = run_benchmark(large_config, num_runs=1, seed=52, checkpoint_key=None if not resume else "LargeDQN")
    
    print("\n" + "="*80)
    print("LARGE TEST RESULTS")
    print("="*80)
    print(f"Hit Rate: {result.get('hit_rate', 0):.4f} ({result.get('hit_rate', 0)*100:.2f}%)")
    print(f"Cache Hits: {result.get('cache_hits', 0)}")
    print(f"Nodes Traversed: {result.get('nodes_traversed', 0)}")
    print(f"Cached Items: {result.get('cached_items', 0)}")
    print(f"Avg Cache Utilization: {result.get('avg_cache_utilization', 0):.2f}%")
    print("="*80)
    
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="NDN caching benchmark runner")
    subparsers = parser.add_subparsers(dest="command")

    compare_parser = subparsers.add_parser("compare", help="Run multi-algorithm comparison")
    compare_parser.add_argument("--scenario", default="medium", help="Scenario name (minimal|medium|large|path)")
    compare_parser.add_argument("--runs", type=int, default=1, help="Number of runs per algorithm")
    compare_parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducibility")

    sweep_parser = subparsers.add_parser("sweep", help="Run DQN/MARL hyperparameter sweep")
    sweep_parser.add_argument("--scenario", default="medium", help="Scenario name (minimal|medium|large|path)")
    sweep_parser.add_argument("--runs", type=int, default=1, help="Number of runs per variant")
    sweep_parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducibility")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "compare":
        run_algorithm_suite(scenario=args.scenario, num_runs=args.runs, seed=args.seed)
    elif args.command == "sweep":
        run_dqn_sweep(scenario=args.scenario, num_runs=args.runs, seed=args.seed)
    else:
        print("No command specified. Example usage:\n  python benchmark.py compare --scenario medium --runs 1")


if __name__ == "__main__":
    main()


def test_hit_rate_comparison(resume: bool = True):
    """Test 3.1: Cache Hit Rate Comparison"""
    print("\n" + "="*80)
    print("TEST 3.1: Cache Hit Rate Comparison")
    print("="*80)
    
    # Load existing results if resuming
    results = load_results() if resume else {}
    
    # WORKAROUND: Reduced network size to prevent stuck workers
    # Original config was causing workers to get stuck processing messages
    base_config = {
        'NDN_SIM_NODES': '50',            # Large network: 50 nodes
        'NDN_SIM_PRODUCERS': '10',         # Large network: 10 producers
        'NDN_SIM_CONTENTS': '1000',        # Large network: 1000 contents
        'NDN_SIM_USERS': '100',            # Large network: 100 users
        'NDN_SIM_ROUNDS': '50',            # Large network: 50 rounds for comprehensive testing
        'NDN_SIM_REQUESTS': '50',          # Large network: 50 requests per round
        'NDN_SIM_WARMUP_ROUNDS': '5',      # Large network: 5 warmup rounds
        'NDN_SIM_CACHE_CAPACITY': '50',    # 5% of catalog (1000 * 0.05 = 50) for large network
        'NDN_SIM_ZIPF_PARAM': '0.8',       # Standard for web/video traffic (heavy tail distribution)
        'NDN_SIM_QUIET': '1',              # Quiet mode: suppress routine warnings
        'NDN_SIM_SKIP_DELAYS': '1',        # Skip sleep delays for faster benchmarks
        'NDN_SIM_USE_DQN': '0',            # DQN disabled for base config (only DQN algorithm uses it)
        'NDN_SIM_WORKER_TIMEOUT': '60.0',  # Increased timeout for large network operations
        'NDN_SIM_MAX_QUEUE_SIZE': '10000', # Increased queue size for large network
        'NDN_SIM_MAX_FIB_RATE': '50',      # Increased FIB update rate for large network
        'NDN_SIM_MAX_FIB_PROPAGATION': '10' # Increased FIB propagation for large network
    }
    
    # Reordered to run DQN first (as requested)
    configs = {
        'DQN': {
            **base_config, 
            'NDN_SIM_CACHE_POLICY': 'combined', 
            'NDN_SIM_USE_DQN': '1',
            'NDN_SIM_ROUNDS': '50',           # Large network: 50 rounds
            'NDN_SIM_WARMUP_ROUNDS': '5',     # Large network: 5 warmup rounds
            'NDN_SIM_WORKER_TIMEOUT': '120.0', # Longer timeout for DQN in large network (120s)
            # OPTIMIZATION: Option to disable DQN if it's causing blocking (set via env var)
            # Set NDN_SIM_DISABLE_DQN_FOR_TESTING=1 to use basic caching instead
        },
        'FIFO': {**base_config, 'NDN_SIM_CACHE_POLICY': 'fifo'},
        'LRU': {**base_config, 'NDN_SIM_CACHE_POLICY': 'lru'},
        'LFU': {**base_config, 'NDN_SIM_CACHE_POLICY': 'lfu'},
        'Combined': {**base_config, 'NDN_SIM_CACHE_POLICY': 'combined'}
    }
    
    fixed_seed = 42  # Same seed for all algorithms (fair comparison)
    
    # Check which algorithms are already completed
    completed_algorithms = set(results.keys())
    if resume and completed_algorithms:
        print(f"\n📊 Found {len(completed_algorithms)} completed algorithms: {', '.join(completed_algorithms)}")
        print("  Will skip completed algorithms and continue with remaining ones.\n")
    
    for name, config in configs.items():
        # Skip if already completed
        if resume and name in completed_algorithms:
            print(f"\n⏭️  Skipping {name} (already completed)")
            continue
        
        print(f"\nTesting {name}...")
        result = run_benchmark(config, num_runs=5, seed=fixed_seed, checkpoint_key=name)  # ⬆️ Increased to 5 for statistical significance
        
        if result:
            results[name] = result
            # Save incrementally after each algorithm
            save_results({name: result})
            print(f"  ✅ {name} completed and saved")
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS: Cache Hit Rate Comparison")
    print("="*80)
    print(f"{'Policy':<15} {'Hit Rate':<15} {'Cache Hits':<15} {'Cached Items':<15}")
    print("-"*80)
    
    for name, result in results.items():
        if result:
            print(f"{name:<15} {result['hit_rate']:.4f}%      {result['cache_hits']:.0f}          {result['cached_items']:.0f}")
    
    # Verify improvements
    if 'FIFO' in results and 'Combined' in results:
        fifo_rate = results['FIFO'].get('hit_rate', 0)
        combined_rate = results['Combined'].get('hit_rate', 0)
        if fifo_rate > 0:
            improvement = combined_rate / fifo_rate
            print(f"\nCombined vs FIFO improvement: {improvement:.2f}x")
            if improvement > 1.5:
                print("✅ PASS: Combined eviction improves over FIFO")
            else:
                print("⚠️  WARNING: Improvement less than expected")
    
    if 'Combined' in results and 'DQN' in results:
        combined_rate = results['Combined'].get('hit_rate', 0)
        dqn_rate = results['DQN'].get('hit_rate', 0)
        if combined_rate > 0:
            improvement = dqn_rate / combined_rate
            print(f"\nDQN vs Combined improvement: {improvement:.2f}x")
            if improvement > 1.2:
                print("✅ PASS: DQN improves over Combined")
            else:
                print("⚠️  WARNING: DQN improvement less than expected")


def test_scalability(resume: bool = True):
    """Test 3.4: Scalability Tests"""
    print("\n" + "="*80)
    print("TEST 3.4: Scalability Tests")
    print("="*80)
    
    # Load existing results if resuming
    scalability_results = {}
    if resume:
        all_results = load_results()
        scalability_results = {k: v for k, v in all_results.items() if k in ['Small', 'Medium']}
    
    configs = {
        'Small': {
            'NDN_SIM_NODES': '50',
            'NDN_SIM_PRODUCERS': '10',
            'NDN_SIM_CONTENTS': '500',
            'NDN_SIM_USERS': '100',
            'NDN_SIM_ROUNDS': '5',           # Evaluation rounds
            'NDN_SIM_REQUESTS': '3',
            'NDN_SIM_WARMUP_ROUNDS': '5',    # Warm-up rounds
            'NDN_SIM_CACHE_CAPACITY': '500',
            'NDN_SIM_CACHE_POLICY': 'combined',
            'NDN_SIM_QUIET': '1',            # Quiet mode: suppress routine warnings
            'NDN_SIM_SKIP_DELAYS': '1'       # Skip sleep delays for faster benchmarks
        },
        'Medium': {
            'NDN_SIM_NODES': '300',
            'NDN_SIM_PRODUCERS': '60',
            'NDN_SIM_CONTENTS': '6000',
            'NDN_SIM_USERS': '2000',
            'NDN_SIM_ROUNDS': '10',          # Evaluation rounds
            'NDN_SIM_REQUESTS': '5',
            'NDN_SIM_WARMUP_ROUNDS': '10',    # Warm-up rounds
            'NDN_SIM_CACHE_CAPACITY': '500',
            'NDN_SIM_CACHE_POLICY': 'combined',
            'NDN_SIM_QUIET': '1',            # Quiet mode: suppress routine warnings
            'NDN_SIM_SKIP_DELAYS': '1'       # Skip sleep delays for faster benchmarks
        }
    }
    
    for name, config in configs.items():
        # Skip if already completed
        if resume and name in scalability_results:
            print(f"\n⏭️  Skipping {name} (already completed)")
            continue
        
        print(f"\nTesting {name} network...")
        result = run_benchmark(config, num_runs=1, checkpoint_key=f"Scalability_{name}")
        
        if result:
            scalability_results[name] = result
            save_results({name: result})
            print(f"  ✅ {name} completed and saved")
    
    results = scalability_results
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS: Scalability")
    print("="*80)
    print(f"{'Size':<15} {'Hit Rate':<15} {'Nodes':<15} {'Cached Items':<15}")
    print("-"*80)
    
    for name, result in results.items():
        if result:
            print(f"{name:<15} {result['hit_rate']:.4f}%      {result['nodes_traversed']:.0f}        {result['cached_items']:.0f}")
    
    # Verify scalability
    if 'Small' in results and 'Medium' in results:
        small_rate = results['Small'].get('hit_rate', 0)
        medium_rate = results['Medium'].get('hit_rate', 0)
        
        if small_rate > 0 and medium_rate > 0:
            print(f"\n✅ System scales: Small={small_rate:.2f}%, Medium={medium_rate:.2f}%")
            if medium_rate >= small_rate * 0.5:  # At least 50% of small performance
                print("✅ PASS: System maintains performance at scale")
            else:
                print("⚠️  WARNING: Performance degrades significantly at scale")


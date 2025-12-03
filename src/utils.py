import threading
import numpy as np
import logging
import warnings
import mmh3
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Set, List, Optional, Deque, Tuple
from datetime import datetime
import random
from collections import deque, OrderedDict
from contextlib import contextmanager
import time

warnings.filterwarnings("ignore")

logger = logging.getLogger('utils_logger')

# Track if semantic encoder warning has been logged (to avoid spam)
_semantic_encoder_import_warned = False

# Simplified NDNDistribution for compatibility with both main.py and run.py
class NDNDistribution:
    """Generates content names following a Zipf-like distribution for NDN simulation"""
    
    def __init__(self, num_contents: int, zipf_param: float = 0.8):
        """
        Initialize content name distribution
        
        Args:
            num_contents: Total number of available contents
            zipf_param: Zipf distribution parameter (controls popularity skew)
        """
        self.num_contents = num_contents
        self.zipf_param = zipf_param
        self.content_list = self.generate_content_names()
        self.probabilities = self._zipf_distribution()
        
        logger.info(f"Initialized NDN distribution with {num_contents} contents")
        
    def generate_content_names(self) -> List[str]:
        """Generate hierarchical content names"""
        content_names = []
        organizations = ['ucla', 'mit', 'stanford', 'berkeley', 'oxford']
        departments = ['cs', 'ee', 'math', 'physics', 'biology']
        content_types = ['research', 'courses', 'projects', 'data', 'media']
        
        for i in range(self.num_contents):
            org = random.choice(organizations)
            dept = random.choice(departments)
            content_type = random.choice(content_types)
            name = f"/edu/{org}/{dept}/{content_type}/content_{i:03d}"
            content_names.append(name)
            
        return content_names
        
    def _zipf_distribution(self) -> np.ndarray:
        """Generate Zipf-like probability distribution for content popularity"""
        ranks = np.arange(1, self.num_contents + 1, dtype=float)
        probs = 1.0 / np.power(ranks, self.zipf_param)
        return probs / probs.sum()
        
    def generate_content_name(self) -> str:
        """Get a random content name based on popularity distribution"""
        return np.random.choice(self.content_list, p=self.probabilities)
        
    def get_popularity(self, content_name: str) -> float:
        """Get popularity score for a content name"""
        try:
            idx = self.content_list.index(content_name)
            return self.probabilities[idx]
        except ValueError:
            return 0.0


# For compatibility with run.py
class ZipfDistribution(NDNDistribution):
    def __init__(self, num_contents, a=0.8):
        super().__init__(num_contents, a)


class ClusterManager:
    def __init__(self, min_cluster_size: int = 2):
        self.min_cluster_size = min_cluster_size
        self.clusters: Dict[int, Set[str]] = {}
        self.content_to_cluster: Dict[str, int] = {}
        self.cluster_scores: Dict[int, float] = {}
        self.cluster_last_access: Dict[int, float] = {}
        self.lock = threading.Lock()
        
    def update_cluster_score(self, cluster_id: int, score_increment: float, current_time: float):
        with self.lock:
            if cluster_id not in self.cluster_scores:
                self.cluster_scores[cluster_id] = 0.0
                self.cluster_last_access[cluster_id] = current_time
                
            time_diff = current_time - self.cluster_last_access.get(cluster_id, 0)
            decay_factor = 0.95 ** max(0, time_diff)  # Ensure non-negative
            self.cluster_scores[cluster_id] = self.cluster_scores[cluster_id] * decay_factor + score_increment
            self.cluster_last_access[cluster_id] = current_time
            
    def get_cluster_score(self, cluster_id: int) -> float:
        with self.lock:
            return self.cluster_scores.get(cluster_id, 0.0)
            
    def get_least_popular_cluster(self) -> int:
        with self.lock:
            if not self.cluster_scores:
                return -1
            return min(self.cluster_scores.items(), key=lambda x: x[1])[0]


class BloomFilter:
    def __init__(self, size: int = 1000, hash_count: int = 4):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = np.zeros(size, dtype=bool)
        self.lock = threading.Lock()
        
    def add(self, item: str):
        with self.lock:
            for seed in range(self.hash_count):
                idx = mmh3.hash(str(item), seed) % self.size
                self.bit_array[idx] = True
                
    def __contains__(self, item: str) -> bool:
        with self.lock:
            return all(
                self.bit_array[mmh3.hash(str(item), seed) % self.size]
                for seed in range(self.hash_count)
            )
            
    def merge(self, other: 'BloomFilter'):
        with self.lock:
            if self.size != other.size:
                raise ValueError("Bloom filters must have the same size")
            self.bit_array |= other.bit_array
    
    def check(self, item: str) -> bool:
        """Check if an item might be in the Bloom filter"""
        return item in self


class NeuralBloomFilter(BloomFilter):
    """
    Task 2.3: Neural Bloom Filter - Enhanced Bloom filter with neural network
    to reduce false positive rates and improve cache state summarization
    
    Uses a small neural network to learn patterns in false positives and
    adjust the filter behavior accordingly.
    """
    def __init__(self, size: int = 2000, hash_count: int = 4, use_neural: bool = True):
        super().__init__(size, hash_count)
        self.use_neural = use_neural
        self.false_positive_history = []  # Track false positives for learning
        self.true_positive_count = 0
        self.false_positive_count = 0
        
        # Task 2.3: Neural network for false positive reduction
        if self.use_neural:
            try:
                import torch
                import torch.nn as nn
                self.torch_available = True
                self._init_neural_network()
            except ImportError:
                self.torch_available = False
                logger.warning("PyTorch not available for NeuralBloomFilter, using basic Bloom filter")
                self.use_neural = False
        else:
            self.torch_available = False
    
    def _init_neural_network(self):
        """Initialize neural network for false positive prediction"""
        try:
            import torch
            import torch.nn as nn
            
            # Small neural network to predict if a Bloom filter check is likely a false positive
            # Input: bit pattern around hash indices, output: probability of false positive
            self.neural_model = nn.Sequential(
                nn.Linear(self.hash_count * 2, 32),  # Input: hash indices and surrounding bits
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),  # Output: false positive probability
                nn.Sigmoid()
            )
            
            # Simple optimizer
            self.optimizer = torch.optim.Adam(self.neural_model.parameters(), lr=0.001)
            self.criterion = nn.BCELoss()
            
            # Training data buffer
            self.training_buffer = []
            self.buffer_size = 100
            
            logger.debug("Neural Bloom filter network initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize neural network for Bloom filter: {e}")
            self.use_neural = False
    
    def _get_bit_pattern(self, item: str) -> np.ndarray:
        """Extract bit pattern around hash indices for neural network input"""
        pattern = np.zeros(self.hash_count * 2, dtype=np.float32)
        for i in range(self.hash_count):
            idx = mmh3.hash(str(item), i) % self.size
            # Get the bit at index and a neighboring bit
            pattern[i * 2] = float(self.bit_array[idx])
            pattern[i * 2 + 1] = float(self.bit_array[(idx + 1) % self.size])
        return pattern
    
    def check(self, item: str, verify_callback=None) -> bool:
        """
        Enhanced check with neural network false positive reduction
        
        Args:
            item: Item to check
            verify_callback: Optional callback to verify if item actually exists
                            (for training the neural network)
        
        Returns:
            True if item might be in filter (with reduced false positive rate)
        """
        # First do basic Bloom filter check
        basic_result = super().check(item)
        
        if not basic_result:
            return False  # Definitely not in filter
        
        # If basic check passes, use neural network to estimate false positive probability
        if self.use_neural and self.torch_available:
            try:
                import torch
                
                # Get bit pattern
                pattern = self._get_bit_pattern(item)
                pattern_tensor = torch.FloatTensor(pattern).unsqueeze(0)
                
                # Get neural network prediction
                with torch.no_grad():
                    false_positive_prob = self.neural_model(pattern_tensor).item()
                
                # If neural network predicts high false positive probability, be more conservative
                # Adjust threshold based on false positive rate
                threshold = 0.5
                if self.false_positive_count + self.true_positive_count > 0:
                    current_fpr = self.false_positive_count / (self.false_positive_count + self.true_positive_count)
                    threshold = max(0.3, min(0.7, 0.5 + current_fpr))
                
                # If neural network suggests high false positive probability, return False
                if false_positive_prob > threshold:
                    return False
                
                # If we have a verification callback, use it to train the network
                if verify_callback is not None:
                    actually_exists = verify_callback(item)
                    if not actually_exists:
                        # False positive - learn from this
                        self.false_positive_count += 1
                        self._train_on_false_positive(pattern, True)
                    else:
                        # True positive
                        self.true_positive_count += 1
                        self._train_on_false_positive(pattern, False)
                
                return True
            except Exception as e:
                logger.debug(f"Neural Bloom filter check failed: {e}, falling back to basic")
                return basic_result
        
        return basic_result
    
    def _train_on_false_positive(self, pattern: np.ndarray, is_false_positive: bool):
        """Train neural network on false positive example"""
        if not self.use_neural or not self.torch_available:
            return
        
        try:
            import torch
            
            # Add to training buffer
            target = 1.0 if is_false_positive else 0.0
            self.training_buffer.append((pattern, target))
            
            # Keep buffer size manageable
            if len(self.training_buffer) > self.buffer_size:
                self.training_buffer.pop(0)
            
            # Train periodically (every 10 examples)
            if len(self.training_buffer) >= 10 and len(self.training_buffer) % 10 == 0:
                # Sample a batch
                batch_size = min(10, len(self.training_buffer))
                batch = self.training_buffer[-batch_size:]
                
                patterns = torch.FloatTensor([p[0] for p in batch])
                targets = torch.FloatTensor([p[1] for p in batch]).unsqueeze(1)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.neural_model(patterns)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                logger.debug(f"Neural Bloom filter trained, loss: {loss.item():.4f}")
        except Exception as e:
            logger.debug(f"Neural Bloom filter training failed: {e}")
    
    def get_false_positive_rate(self) -> float:
        """Get current estimated false positive rate"""
        total = self.false_positive_count + self.true_positive_count
        if total == 0:
            return 0.0
        return self.false_positive_count / total
    
    def reset_stats(self):
        """Reset false positive statistics"""
        self.false_positive_count = 0
        self.true_positive_count = 0
        self.training_buffer.clear()


class ContentStore:
    def __init__(self, total_capacity: int, router_id: int):
        self.store: Dict[str, Any] = {}
        self.size_map: Dict[str, int] = {}
        self.router_id = router_id
        self.remaining_capacity = total_capacity
        self.total_capacity = total_capacity
        self.store_order: Deque[str] = deque()
        self.insertions = 0
        self._producer_hop_cache: Dict[int, int] = {}
        self._pending_inference_batch: List[Tuple[np.ndarray, Dict[str, int], threading.Event]] = []
        self._batch_lock = threading.Lock()
        self._batch_timer: Optional[threading.Timer] = None
        self.dqn_batch_trigger = int(os.environ.get("NDN_SIM_DQN_BATCH_TRIGGER", "4"))
        self.dqn_batch_delay = float(os.environ.get("NDN_SIM_DQN_BATCH_DELAY", "0.003"))
        self._decision_cache: Dict[str, Tuple[int, float]] = {}
        self.decision_cache_ttl = float(os.environ.get("NDN_SIM_DECISION_CACHE_TTL", "0.05"))
        self.twolru_filter_capacity = int(
            os.environ.get("NDN_SIM_TWOLRU_FILTER_SIZE", str(max(10, total_capacity)))
        )
        self.twolru_filter: "OrderedDict[str, float]" = OrderedDict()
        
        # Simplified semantic clustering components
        self.cluster_manager = ClusterManager()
        self.content_embeddings: Dict[str, np.ndarray] = {}
        
        # Task 2.2: Enhanced semantic encoding with neural networks
        global _semantic_encoder_import_warned
        try:
            from semantic_encoder import get_semantic_encoder
            self.semantic_encoder = get_semantic_encoder(embedding_dim=64, use_cnn=True)
            if not _semantic_encoder_import_warned:
                logger.info(f"ContentStore {router_id}: Using CNN-based semantic encoder")
        except Exception as e:
            if not _semantic_encoder_import_warned:
                logger.warning(f"Failed to load semantic encoder: {e}, using hash-based for all ContentStores")
                _semantic_encoder_import_warned = True
            self.semantic_encoder = None
        
        # Thread safety
        self.store_lock = threading.Lock()
        self.embedding_lock = threading.Lock()
        
        # Task 2.3: Use Neural Bloom Filter for better cache state summarization
        # Phase 3.2: Adaptive Bloom filter sizing based on cache capacity and false positive tolerance
        try:
            use_neural_bloom = os.environ.get("NDN_SIM_NEURAL_BLOOM", "0") == "1"
            
            # Phase 3.2: Calculate optimal Bloom filter size
            # Optimal size: m = -n * ln(p) / (ln(2)^2)
            # where n = expected cache size, p = desired false positive rate
            desired_fpr = float(os.environ.get("NDN_SIM_BLOOM_FPR", "0.01"))  # Default 1% FPR
            expected_cache_size = max(10, total_capacity)  # Use cache capacity as expected size
            
            # Calculate optimal size
            import math
            if expected_cache_size > 0 and desired_fpr > 0:
                optimal_size = int(-expected_cache_size * math.log(desired_fpr) / (math.log(2) ** 2))
                # Round to nearest 100 for efficiency
                optimal_size = ((optimal_size + 50) // 100) * 100
                # Clamp to reasonable range [500, 10000]
                optimal_size = max(500, min(10000, optimal_size))
                
                # Calculate optimal hash count: k = (m/n) * ln(2)
                optimal_hash_count = max(2, min(8, int((optimal_size / max(1, expected_cache_size)) * math.log(2))))
            else:
                # Fallback to defaults
                optimal_size = 2000
                optimal_hash_count = 4
            
            if use_neural_bloom:
                self.bloom_filter = NeuralBloomFilter(size=optimal_size, hash_count=optimal_hash_count, use_neural=True)
                logger.info(f"ContentStore {router_id}: Using Neural Bloom Filter (size={optimal_size}, hash_count={optimal_hash_count}, FPR={desired_fpr:.3f})")
            else:
                self.bloom_filter = BloomFilter(size=optimal_size, hash_count=optimal_hash_count)
                logger.debug(f"ContentStore {router_id}: Using Bloom Filter (size={optimal_size}, hash_count={optimal_hash_count}, FPR={desired_fpr:.3f})")
        except Exception as e:
            logger.warning(f"ContentStore {router_id}: Failed to initialize Bloom Filter with adaptive sizing: {e}, using defaults")
            self.bloom_filter = BloomFilter(size=2000, hash_count=4)
        
        self.neighbor_filters: Dict[int, BloomFilter] = {}
        
        # FeiWang mode: Exact neighbor cache state (for exact neighbor state exchange)
        self.neighbor_exact_states: Dict[int, set] = {}  # neighbor_id -> set of cached content names
        self.exact_state_propagation_frequency = 10  # Propagate every N cache insertions
        self.last_exact_state_propagation = 0
        
        # Task 2.4: Store references to router and graph for enhanced DQN state
        self.router_ref = None  # Will be set by router after initialization
        self.graph_ref = None  # Will be set by router after initialization
        
        # Access tracking
        self.access_count: Dict[str, int] = {}
        self.last_access_time: Dict[str, float] = {}
        
        # DQN tracking: track which contents were cached by DQN decisions
        self.dqn_cached_contents: Set[str] = set()
        self.dqn_decision_states: Dict[str, np.ndarray] = {}  # Track state when caching decision was made
        
        # DQN training frequency control
        self.dqn_training_step = 0
        # Allow training frequency to be configured via environment variable
        # OPTIMIZED: More frequent training for faster learning (10 -> 5)
        self.dqn_training_frequency = int(os.environ.get('NDN_SIM_DQN_TRAINING_FREQUENCY', os.environ.get('DQN_TRAINING_FREQUENCY', '5')))
        
        # Bloom filter propagation control
        self.bloom_propagation_frequency = 10  # Propagate every N cache insertions
        self.last_bloom_propagation = 0
        
        # Cache embeddings for semantic similarity (feature 15)
        self.cached_embeddings: Dict[str, np.ndarray] = {}
        
        # For compatibility with run.py
        self.mode = "basic"  # Options: "basic", "dqn_cache", "opt", "lfo", "fei_wang"
        self.status = "router"
        self.dqn_agent = None
        self.replacement_policy = "lru"  # Options: fifo, lifo, lru, basic
        
        # Baseline caching support
        self.baseline_cache = None  # Will hold OptimalCaching, LFOBaseline, FeiWangICC2023Baseline
        
        # Initialize DQN agent if mode is set to use it
        if self.mode == "dqn_cache":
            self.initialize_dqn_agent()
        
        logger.info(f"Initialized ContentStore for router {router_id} with capacity {total_capacity}")
    
    def debug_cache_contents(self):
        """Debug method to print cache contents"""
        with self.store_lock:
            print(f"ContentStore {self.router_id} contents:")
            print(f"  Capacity: {self.total_capacity}, Used: {self.total_capacity - self.remaining_capacity}")
            print(f"  Items cached: {len(self.store)}")
            for name, content in list(self.store.items())[:5]:  # Show first 5 items
                size = self.size_map.get(name, 0)
                accesses = self.access_count.get(name, 0)
                print(f"    - {name} (Size: {size}, Accesses: {accesses})")
            if len(self.store) > 5:
                print(f"    ... and {len(self.store) - 5} more items")

    def fix_caching_policies(self):
        """Fix and improve caching policies"""
        # First, make sure we have room for at least a few items
        with self.store_lock:
            # Check if we need to clear some space
            if self.remaining_capacity < (self.total_capacity * 0.2):
                # Remove least accessed items to free up 30% of capacity
                target_space = self.total_capacity * 0.3
                
                # Sort by access count (ascending)
                items_to_check = sorted(
                    self.store.keys(),
                    key=lambda x: self.access_count.get(x, 0)
                )
                
                space_freed = 0
                for name in items_to_check:
                    if space_freed >= target_space:
                        break
                    size = self.size_map.get(name, 0)
                    self.remove_content(name)
                    space_freed += size
                    
                print(f"ContentStore {self.router_id}: Cleared {space_freed} units to make room")
        
        # Make sure the mode is properly set
        if hasattr(self, 'mode') and self.mode != "dqn_cache":
            self.mode = "dqn_cache"
            # Initialize DQN agent if needed
            if hasattr(self, 'initialize_dqn_agent') and self.dqn_agent is None:
                try:
                    self.initialize_dqn_agent()
                except Exception as e:
                    print(f"Could not initialize DQN agent: {e}")
    
    def initialize_dqn_agent(self):
        """Initialize the DQN agent for caching decisions"""
        try:
            # Import here to avoid circular imports
            from dqn_agent import DQNAgent
            
            # Optimized state dimensions include local cache status, neighbor utilization, hop distance, and demand
            state_dim = 8
            # Action dimensions: 0 = don't cache, 1 = cache
            action_dim = 2
            
            # Create the agent with consistent hyperparameters matching DQNAgent defaults
            # OPTIMIZED: Allow batch size to be configured via environment variable for GPU optimization
            batch_size = int(os.environ.get('DQN_BATCH_SIZE', '128'))  # OPTIMIZED: 64 -> 128
            lr = float(os.environ.get('NDN_SIM_DQN_LR', '0.0005'))  # Default: 0.0005, optimized: 0.001
            gamma = float(os.environ.get('NDN_SIM_DQN_GAMMA', '0.995'))  # OPTIMIZED: 0.99 -> 0.995
            epsilon_start = float(os.environ.get('NDN_SIM_DQN_EPSILON_START', '1.0'))
            epsilon_end = float(os.environ.get('NDN_SIM_DQN_EPSILON_END', '0.05'))  # Default: 0.05, optimized: 0.02
            epsilon_decay = float(os.environ.get('NDN_SIM_DQN_EPSILON_DECAY', '0.9995'))  # Default: 0.9995, optimized: 0.998
            target_update_freq = int(os.environ.get('NDN_SIM_DQN_TARGET_UPDATE', '200'))  # OPTIMIZED: 100 -> 200
            n_step = int(os.environ.get('NDN_SIM_DQN_N_STEP', '30'))  # OPTIMIZED: 20 -> 30
            memory_size_env = os.environ.get('NDN_SIM_DQN_MEMORY_SIZE')
            if memory_size_env:
                memory_size = int(memory_size_env)
            else:
                memory_multiplier = float(os.environ.get('NDN_SIM_DQN_MEMORY_MULTIPLIER', '25'))  # Default: 25, optimized: 50
                memory_size = max(10000, int(self.total_capacity * memory_multiplier))
            hidden_dims_env = os.environ.get('NDN_SIM_DQN_HIDDEN_DIMS', '').strip()
            if hidden_dims_env:
                hidden_dims = [int(dim.strip()) for dim in hidden_dims_env.split(',') if dim.strip()]
            else:
                hidden_dims = [512, 256, 128]  # OPTIMIZED: [256, 128, 64] -> [512, 256, 128]
            experiment_id_env = os.environ.get('NDN_SIM_EXPERIMENT_ID')
            tensorboard_dir_env = os.environ.get('NDN_SIM_TENSORBOARD_DIR')
            self.dqn_agent = DQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                memory_size=memory_size,  # Increased for better stability
                batch_size=batch_size,  # Configurable via DQN_BATCH_SIZE env var
                gamma=gamma,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                epsilon_decay=epsilon_decay,
                lr=lr,
                target_update_freq=target_update_freq,
                n_step=n_step,
                hidden_dims=hidden_dims,
                router_id=getattr(self, "router_id", None),
                experiment_id=experiment_id_env,
                tensorboard_dir=tensorboard_dir_env
            )
            
            logger.info(f"DQN agent initialized for router {self.router_id} with state_dim={state_dim}")
            return True
        except Exception as e:
            logger.error(f"Error initializing DQN agent: {e}")
            # Fall back to basic caching if DQN fails
            self.mode = "basic"
            return False
    
    def set_mode(self, mode: str):
        """Set caching policy mode"""
        if mode == self.mode:
            return
            
        old_mode = self.mode
        self.mode = mode
        
        if mode == "dqn_cache" and self.dqn_agent is None:
            # Task 1.3: Initialize DQN agent when switching to DQN mode
            success = self.initialize_dqn_agent()
            if not success:
                logger.warning(f"ContentStore {self.router_id}: Failed to initialize DQN agent, falling back to basic mode")
                self.mode = "basic"
            else:
                logger.info(f"ContentStore {self.router_id}: Switched from {old_mode} to DQN caching mode")
        elif mode == "opt":
            # OPT (Optimal Offline) baseline using Belady's algorithm
            try:
                from baselines import OptimalCaching
                self.baseline_cache = OptimalCaching(self.router_id, self.total_capacity)
                logger.info(f"ContentStore {self.router_id}: Switched to OPT (Optimal Offline) baseline mode")
            except Exception as e:
                logger.warning(f"ContentStore {self.router_id}: Failed to initialize OPT baseline: {e}, falling back to basic")
                self.mode = "basic"
        elif mode == "lfo":
            # LFO (Least Frequently Optimal) baseline
            try:
                from baselines import LFOBaseline
                self.baseline_cache = LFOBaseline(self.router_id, self.total_capacity)
                logger.info(f"ContentStore {self.router_id}: Switched to LFO baseline mode")
            except Exception as e:
                logger.warning(f"ContentStore {self.router_id}: Failed to initialize LFO baseline: {e}, falling back to basic")
                self.mode = "basic"
        elif mode == "fei_wang":
            # Fei Wang ICC 2023 baseline
            try:
                from baselines import FeiWangICC2023Baseline
                self.baseline_cache = FeiWangICC2023Baseline(self.router_id, self.total_capacity)
                logger.info(f"ContentStore {self.router_id}: Switched to Fei Wang ICC 2023 baseline mode")
            except Exception as e:
                logger.warning(f"ContentStore {self.router_id}: Failed to initialize Fei Wang baseline: {e}, falling back to basic")
                self.mode = "basic"
        elif mode != "dqn_cache":
            # Fall back to requested replacement policy, keep existing choice
            self.mode = "basic"
            logger.debug(f"ContentStore {self.router_id}: Using basic caching mode ({mode})")

    def set_replacement_policy(self, policy: str):
        """Configure replacement policy used in basic mode caches."""
        normalized = policy.lower()
        # Task 2.1: Add "combined" and "lfu" to supported policies
        # Also support baseline policies: "opt", "lfo", "fei_wang"
        if normalized in {"fifo", "lifo", "lru", "lfu", "basic", "combined", "2-lru", "opt", "lfo", "fei_wang"}:
            # If it's a baseline policy, set mode instead of replacement_policy
            if normalized in {"opt", "lfo", "fei_wang"}:
                self.set_mode(normalized)
            else:
                self.replacement_policy = normalized
                logger.info(f"ContentStore {self.router_id}: Replacement policy set to {normalized}")
        else:
            logger.warning(f"ContentStore {self.router_id}: Unknown replacement policy {policy}, keeping {self.replacement_policy}")

    def _twolru_should_admit(self, content_name: str) -> bool:
        """
        Admission control for 2-LRU policy.
        Returns True on second request (allow cache), False on first (populate filter only).
        """
        if content_name in self.twolru_filter:
            self.twolru_filter.pop(content_name)
            self.twolru_filter[content_name] = time.time()
            return True
        self.twolru_filter[content_name] = time.time()
        if len(self.twolru_filter) > self.twolru_filter_capacity:
            self.twolru_filter.popitem(last=False)
        return False
    
    def generate_embedding(self, name: str) -> np.ndarray:
        """
        Task 2.2: Generate semantic embedding using CNN-based encoder or hash fallback
        """
        # Use semantic encoder if available
        if hasattr(self, 'semantic_encoder') and self.semantic_encoder is not None:
            try:
                return self.semantic_encoder.encode(name)
            except Exception as e:
                logger.warning(f"Semantic encoder failed for {name}: {e}, using hash fallback")
        
        # Fallback to hash-based embedding (compatible with existing code)
        parts = name.split('/')
        embedding = np.zeros(64, dtype=np.float32)  # Match semantic encoder dimension
        for i, part in enumerate(parts[:min(len(parts), 64)]):
            # Use hash to generate a pseudo-random embedding
            hash_val = mmh3.hash(part, i) % 1000
            embedding[i % 64] = hash_val / 1000.0
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
            
    def update_clusters(self, new_content: str, current_time: float):
        """Update content clusters - simplified version"""
        with self.embedding_lock:
            # Generate embedding for new content if needed
            if new_content not in self.content_embeddings:
                self.content_embeddings[new_content] = self.generate_embedding(new_content)
            
            # Simple clustering based on content name prefixes
            parts = new_content.split('/')
            if len(parts) >= 4:
                # Use domain prefix as cluster
                domain_prefix = '/'.join(parts[:4])
                cluster_id = abs(hash(domain_prefix) % 100)  # Use hash value as cluster ID
                
                # Update cluster membership
                if cluster_id not in self.cluster_manager.clusters:
                    self.cluster_manager.clusters[cluster_id] = set()
                    
                # Add to new cluster
                self.cluster_manager.clusters[cluster_id].add(new_content)
                self.cluster_manager.content_to_cluster[new_content] = cluster_id
                
                # Update cluster score
                access_score = self.access_count.get(new_content, 0)
                self.cluster_manager.update_cluster_score(cluster_id, access_score, current_time)
            
    def evict_from_least_popular_cluster(self, required_space: int) -> bool:
        """Evict content from least popular cluster, with fallback to LRU if cluster doesn't have enough"""
        least_popular_cluster = self.cluster_manager.get_least_popular_cluster()
        if least_popular_cluster == -1:
            # Fall back to LRU if no clusters exist
            return self.evict_lru(required_space)
            
        # Get content in the least popular cluster
        cluster_contents = self.cluster_manager.clusters.get(least_popular_cluster, set())
        if not cluster_contents:
            return self.evict_lru(required_space)
            
        # Sort by access count and time
        sorted_contents = sorted(
            [c for c in cluster_contents if c in self.store],
            key=lambda x: (self.access_count.get(x, 0), -self.last_access_time.get(x, 0))
        )
        
        space_freed = 0
        for content in sorted_contents:
            if content in self.store:
                space_freed += self.size_map.get(content, 0)
                self._remove_content_no_lock(content)
                
                if space_freed >= required_space:
                    return True
        
        # CRITICAL FIX: If cluster-based eviction didn't free enough space, fall back to LRU
        # This prevents cache failures when the least popular cluster is too small
        if space_freed < required_space:
            remaining_needed = required_space - space_freed
            return self.evict_lru(remaining_needed)
                    
        return space_freed >= required_space
    
    def combined_eviction_algorithm(self, required_space: int, weight: float = 0.5, current_time: float = None) -> bool:
        """
        Task 2.1: Algorithm 1 - Combined Eviction Algorithm (recency + frequency)
        From research report: combines recency_scores and frequency_scores
        
        Args:
            required_space: Space needed to free up
            weight: Weight for recency (0.0-1.0), frequency weight = 1-weight
            current_time: Current time for recency calculation (default: now)
        
        Returns:
            True if enough space was freed, False otherwise
        """
        if not self.store:
            return False
        
        if current_time is None:
            import time
            current_time = time.time()
        
        # Calculate recency_scores (inverse of time since last access)
        recency_scores = {}
        max_time_diff = 0.0
        for name in self.store:
            last_access = self.last_access_time.get(name, 0)
            time_diff = max(0.1, current_time - last_access)  # Avoid division by zero
            recency_scores[name] = 1.0 / time_diff  # Higher score = more recent
            max_time_diff = max(max_time_diff, time_diff)
        
        # Normalize recency scores to 0-1 range
        if max_time_diff > 0:
            for name in recency_scores:
                recency_scores[name] = recency_scores[name] / (1.0 / 0.1)  # Normalize
        
        # Calculate frequency_scores (based on access_count)
        frequency_scores = {}
        max_frequency = max(self.access_count.values()) if self.access_count else 1.0
        for name in self.store:
            freq = self.access_count.get(name, 0)
            frequency_scores[name] = freq / max(1.0, max_frequency)  # Normalize to 0-1
        
        # Combine scores: combined_scores = weight * recency + (1-weight) * frequency
        # Lower combined score = less important = candidate for eviction
        combined_scores = {}
        for name in self.store:
            recency = recency_scores.get(name, 0.0)
            frequency = frequency_scores.get(name, 0.0)
            combined_scores[name] = weight * recency + (1.0 - weight) * frequency
        
        # Sort by combined score (ascending) - lowest scores first (evict these)
        sorted_contents = sorted(
            self.store.keys(),
            key=lambda x: combined_scores.get(x, 0.0)
        )
        
        # Evict content with lowest combined scores until we have enough space
        space_freed = 0
        for content_name in sorted_contents:
            if space_freed >= required_space:
                break
            content_size = self.size_map.get(content_name, 0)
            space_freed += content_size
            self._remove_content_no_lock(content_name)
            logger.debug(
                f"ContentStore {self.router_id}: Evicted {content_name} "
                f"(score={combined_scores.get(content_name, 0):.3f}, "
                f"recency={recency_scores.get(content_name, 0):.3f}, "
                f"frequency={frequency_scores.get(content_name, 0):.3f})"
            )
        
        return space_freed >= required_space
    
    def evict_lru(self, required_space: int) -> bool:
        """Evict content using LRU policy"""
        if not self.store:
            return False
            
        # Sort content by last access time
        lru_contents = sorted(
            self.store.keys(),
            key=lambda x: self.last_access_time.get(x, 0)
        )
        
        space_freed = 0
        for content in lru_contents:
            space_freed += self.size_map.get(content, 0)
            self._remove_content_no_lock(content)
            
            if space_freed >= required_space:
                return True
                
        return space_freed >= required_space
    
    def evict_lfu(self, required_space: int) -> bool:
        """Evict content using LFU (Least Frequently Used) policy"""
        space_freed = 0
        if not self.store:
            return False
        
        # Sort by access count (ascending) - least frequent first
        lfu_contents = sorted(
            self.store.keys(),
            key=lambda x: self.access_count.get(x, 0)
        )
        
        for content_name in lfu_contents:
            if space_freed >= required_space:
                break
            space_freed += self.size_map.get(content_name, 0)
            self._remove_content_no_lock(content_name)
                
        return space_freed >= required_space
    
    def evict_fifo(self, required_space: int) -> bool:
        """Evict content using FIFO policy."""
        space_freed = 0
        while self.store_order and space_freed < required_space:
            oldest = self.store_order.popleft()
            if oldest in self.store:
                space_freed += self.size_map.get(oldest, 0)
                self._remove_content_no_lock(oldest)
        return space_freed >= required_space

    def evict_lifo(self, required_space: int) -> bool:
        """Evict content using LIFO policy."""
        space_freed = 0
        while self.store_order and space_freed < required_space:
            newest = self.store_order.pop()
            if newest in self.store:
                space_freed += self.size_map.get(newest, 0)
                self._remove_content_no_lock(newest)
        return space_freed >= required_space

    def evict_basic(self, required_space: int) -> bool:
        """Default heuristic eviction (cluster-based fallback)."""
        return self.evict_from_least_popular_cluster(required_space)
    
    def evict_by_policy(self, required_space: int, current_time: float = None) -> bool:
        """Dispatch eviction based on configured policy."""
        policy = self.replacement_policy
        result = False
        
        if policy == "fifo":
            result = self.evict_fifo(required_space)
        elif policy == "lifo":
            result = self.evict_lifo(required_space)
        elif policy == "lru":
            result = self.evict_lru(required_space)
        elif policy == "2-lru":
            result = self.evict_lru(required_space)
        elif policy == "lfu":
            result = self.evict_lfu(required_space)
        elif policy == "combined":
            # Task 2.1: Use combined eviction algorithm (Algorithm 1 from report)
            result = self.combined_eviction_algorithm(required_space, weight=0.5, current_time=current_time)
        else:
        # "basic" or unknown - default to combined algorithm
            result = self.combined_eviction_algorithm(required_space, weight=0.5, current_time=current_time)
    
        # CRITICAL FIX 2: Defensive fallback - ensure eviction succeeds if cache has items
        # If policy-based eviction failed but cache has items, use aggressive LRU eviction
        if not result and len(self.store) > 0:
            # Last resort: evict oldest items until we have enough space
            space_freed = 0
            items_to_evict = []
            
            # Sort by last access time (oldest first) - simple LRU fallback
            sorted_items = sorted(
                self.store.keys(),
                key=lambda x: self.last_access_time.get(x, 0)
            )
            
            for item in sorted_items:
                item_size = self.size_map.get(item, 0)
                items_to_evict.append(item)
                space_freed += item_size
                if space_freed >= required_space:
                    break
            
            # Evict collected items
            for item in items_to_evict:
                self._remove_content_no_lock(item)
            
            result = space_freed >= required_space
            if result:
                logger.debug(
                    f"ContentStore {self.router_id}: Defensive eviction succeeded "
                    f"(freed={space_freed}, required={required_space}, evicted={len(items_to_evict)} items)"
                )
        
        return result
    
    def _estimate_hop_distance_to_producer(self) -> int:
        if self.router_ref is None or self.graph_ref is None:
            return 0
        router_id = self.router_ref.router_id
        if router_id in self._producer_hop_cache:
            return self._producer_hop_cache[router_id]
        try:
            import networkx as nx
            producer_nodes = [
                node for node, data in self.graph_ref.nodes(data=True)
                if data.get('type') == 'producer'
            ]
            if not producer_nodes:
                return 0
            hop = min(
                nx.shortest_path_length(self.graph_ref, router_id, producer)
                for producer in producer_nodes
            )
            self._producer_hop_cache[router_id] = hop
            return hop
        except Exception:
            return 0

    def _process_inference_batch(self, entries: List[Tuple[np.ndarray, Dict[str, int], threading.Event]]):
        if self.dqn_agent is None or not entries:
            for _, result_holder, event in entries:
                result_holder['action'] = 0
                event.set()
            return
        try:
            states = [entry[0] for entry in entries]
            actions = self.dqn_agent.select_action_batch(states)
            if len(actions) != len(entries):
                actions = [self.dqn_agent.select_action(state) for state in states]
        except Exception as e:
            logger.debug(f"ContentStore {self.router_id}: Error during batch inference: {e}")
            actions = [self.dqn_agent.select_action(entry[0]) for entry in entries]
        for (_, result_holder, event), action in zip(entries, actions):
            result_holder['action'] = action
            event.set()

    def _flush_inference_batch(self):
        entries: List[Tuple[np.ndarray, Dict[str, int], threading.Event]] = []
        with self._batch_lock:
            if self._pending_inference_batch:
                entries = self._pending_inference_batch
                self._pending_inference_batch = []
            self._batch_timer = None
        if entries:
            self._process_inference_batch(entries)

    def _select_action_with_batching(self, state: np.ndarray) -> int:
        if self.dqn_agent is None:
            return 0
        if self.dqn_batch_trigger <= 1:
            return self.dqn_agent.select_action(state)
        result_holder: Dict[str, int] = {}
        done_event = threading.Event()
        entries_to_process = None
        with self._batch_lock:
            self._pending_inference_batch.append((state, result_holder, done_event))
            if len(self._pending_inference_batch) >= self.dqn_batch_trigger:
                entries_to_process = self._pending_inference_batch
                self._pending_inference_batch = []
                if self._batch_timer:
                    self._batch_timer.cancel()
                    self._batch_timer = None
            elif self._batch_timer is None:
                self._batch_timer = threading.Timer(self.dqn_batch_delay, self._flush_inference_batch)
                self._batch_timer.daemon = True
                self._batch_timer.start()
        if entries_to_process:
            self._process_inference_batch(entries_to_process)
        done_event.wait(timeout=0.1)
        action = result_holder.get('action')
        if action is None:
            action = self.dqn_agent.select_action(state)
        return action
                
    def get_state_for_dqn(self, content_name: str, content_size: int, router=None, G=None, current_time: float = None, lock_already_held: bool = False) -> np.ndarray:
        """
        Optimized DQN state space with blended local, neighbor, and demand features (8 features)
        Removed redundant features: cluster score, node degree, semantic similarity, content popularity, cache utilization.
        
        Key feature: Feature 4 (neighbor has content via Bloom filters) - enables
        distributed coordination without central control.
        
        Args:
            content_name: Name of content
            content_size: Size of content
            router: Router reference (optional, uses self.router_ref if None)
            G: Network graph (optional, uses self.graph_ref if None)
            current_time: Current simulation time (optional)
            lock_already_held: If True, assumes store_lock is already held and skips acquiring it
        
        Returns:
            8-dimensional state vector:
            [0] Content already cached (binary)
            [1] Content size (normalized)
            [2] Remaining capacity (normalized)
            [3] Access frequency (normalized)
            [4] Neighbor has content via Bloom filters
            [5] Average neighbor utilization
            [6] Hop closeness to producer (1 / (1 + hop))
            [7] Router demand / PIT load
        """
        # Error handling: Validate inputs
        if router is None:
            router = self.router_ref
        if G is None:
            G = self.graph_ref
        if current_time is None and hasattr(self, 'router_ref') and self.router_ref:
            current_time = getattr(self.router_ref, 'router_time', 0.0)
        
        # CRITICAL DEADLOCK FIX: Get neighbors FIRST, outside all locks
        # Acquiring neighbor_lock while holding store_lock causes deadlocks
        neighbors = []
        if router is not None and hasattr(router, 'neighbors'):
            try:
                # Get neighbors with its own lock (NOT while holding store_lock)
                if hasattr(router, 'neighbor_lock'):
                    with router.neighbor_lock:
                        neighbors = list(router.neighbors) if hasattr(router, 'neighbors') else []
                else:
                    neighbors = list(router.neighbors) if hasattr(router, 'neighbors') else []
            except (AttributeError, Exception) as e:
                logger.debug(f"ContentStore {self.router_id}: Error accessing neighbors: {e}")
                neighbors = []
        
        # Get neighbor_filters snapshot outside lock (read-only, thread-safe dict)
        neighbor_filters_snapshot = dict(self.neighbor_filters) if hasattr(self, 'neighbor_filters') else {}
        
        # Now acquire store_lock for fast cache state reads
        # CRITICAL: We already have neighbors, so we won't deadlock trying to get neighbor_lock
        if lock_already_held:
            # Lock already held by caller, use no-op context manager
            @contextmanager
            def noop_lock():
                yield
            lock_context = noop_lock()
        else:
            lock_context = self.store_lock
        
        with lock_context:
            # Optimized state: 5 essential features (removed Feature 3: cache utilization - redundant with Feature 2)
            state = np.zeros(8, dtype=np.float32)
            
            # Feature 0: Content already cached (binary)
            state[0] = float(content_name in self.store)
            
            # Feature 1: Content size (normalized)
            state[1] = float(content_size) / max(1, self.total_capacity)
            
            # Feature 2: Remaining cache capacity (normalized)
            state[2] = self.remaining_capacity / max(1, self.total_capacity)
            
            # Feature 3: Access frequency (normalized)
            access_count = self.access_count.get(content_name, 0)
            total_accesses = sum(self.access_count.values()) if self.access_count else 1
            state[3] = access_count / max(1, total_accesses)
            
        # CRITICAL FIX: Do slow graph operations OUTSIDE the lock
        # _calculate_neighbor_importance calls nx.shortest_path_length which can be slow
        if neighbors:
            # FeiWang mode: Support for exact neighbor state exchange (ICC 2023)
            use_exact_neighbor = os.environ.get("NDN_SIM_USE_EXACT_NEIGHBOR_STATE", "0") == "1"
            
            if use_exact_neighbor:
                # FeiWang mode: Use exact neighbor cache contents (exact match, not Bloom filter)
                # Count neighbors that actually have the content (exact match)
                neighbor_has_content = 0
                neighbor_exact_states_snapshot = dict(self.neighbor_exact_states) if hasattr(self, 'neighbor_exact_states') else {}
                
                for neighbor_id in neighbors:
                    neighbor_cache = neighbor_exact_states_snapshot.get(neighbor_id, set())
                    if content_name in neighbor_cache:
                        neighbor_has_content += 1
                
                # Fraction of neighbors that have the content (exact)
                state[4] = neighbor_has_content / max(1, len(neighbors))
            else:
                # Phase 7.1: Support for ablation study - disable Bloom filter feature
                disable_bloom = os.environ.get("NDN_SIM_DISABLE_BLOOM", "0") == "1"
                
                if disable_bloom:
                    # Ablation variant: DQN without Bloom filters (no neighbor awareness)
                    # Set Feature 4 to 0.0 (no neighbor information)
                    state[4] = 0.0
                else:
                    # Phase 3.1: Adaptive neighbor selection (weighted by importance)
                    # Weight neighbors by: traffic volume, distance, hit rate
                    # CRITICAL FIX: This is the slow part - graph algorithms - do it outside lock
                    # OPTIMIZATION: Skip neighbor importance calculation if it's too slow (use equal weights)
                    # This prevents workers from getting stuck on slow graph operations
                    try:
                        neighbor_weights = self._calculate_neighbor_importance(neighbors, router, G)
                    except Exception as e:
                        # If calculation fails or is too slow, use equal weights (fast fallback)
                        logger.debug(f"ContentStore {self.router_id}: Neighbor importance calculation failed: {e}, using equal weights")
                        neighbor_weights = {neighbor_id: 1.0 for neighbor_id in neighbors}
                    
                    # Weighted count of neighbors that might have content (via Bloom filters)
                    weighted_neighbor_has_content = 0.0
                    total_weight = 0.0
                    
                    for neighbor_id in neighbors:
                        neighbor_filter = neighbor_filters_snapshot.get(neighbor_id)
                        if neighbor_filter is not None:
                            # Check if neighbor's Bloom filter indicates content might be cached
                            neighbor_has = 1.0 if neighbor_filter.check(content_name) else 0.0
                            weight = neighbor_weights.get(neighbor_id, 1.0)  # Default weight = 1.0
                            weighted_neighbor_has_content += neighbor_has * weight
                            total_weight += weight
                    
                    # Weighted fraction of neighbors that might have content
                    state[4] = weighted_neighbor_has_content / max(1.0, total_weight) if total_weight > 0 else 0.0
        else:
            state[4] = 0.0
        
        # Feature 5: Neighbor utilization
        neighbor_util = 0.0
        neighbor_samples = 0
        graph_ref = G or self.graph_ref
        if neighbors and graph_ref is not None:
            for neighbor_id in neighbors:
                try:
                    neighbor_router = graph_ref.nodes[neighbor_id].get('router')
                except Exception:
                    neighbor_router = None
                if neighbor_router and hasattr(neighbor_router, 'content_store'):
                    n_cs = neighbor_router.content_store
                    total = getattr(n_cs, 'total_capacity', 0)
                    if total > 0:
                        used = total - getattr(n_cs, 'remaining_capacity', 0)
                        neighbor_util += used / total
                        neighbor_samples += 1
        state[5] = neighbor_util / max(1, neighbor_samples)
        
        # Feature 6: Hop closeness to nearest producer
        hop_distance = self._estimate_hop_distance_to_producer()
        state[6] = 1.0 / (1.0 + hop_distance)
        
        # Feature 7: PIT load / demand indicator
        pit_entries = getattr(getattr(router, 'PIT', None), 'entries', {}) if router else {}
        pit_load = len(pit_entries) if pit_entries else 0
        state[7] = min(1.0, pit_load / max(1, len(neighbors) + 1))
        
        return state
    
    def _calculate_neighbor_importance(self, neighbors: List[int], router=None, G=None) -> Dict[int, float]:
        """
        Phase 3.1: Calculate importance weights for neighbors
        
        Weights neighbors by:
        - Traffic volume (number of messages exchanged)
        - Distance (hop count in network)
        - Hit rate (if available)
        
        Args:
            neighbors: List of neighbor IDs
            router: Router reference (optional)
            G: Network graph (optional)
        
        Returns:
            Dictionary mapping neighbor_id -> importance weight (0.0 to 1.0)
        """
        # CRITICAL OPTIMIZATION: Skip slow graph operations entirely to prevent worker blocking
        # The shortest_path_length calculation can be very slow and blocks workers
        # Use equal weights as a fast fallback - this maintains functionality while avoiding deadlocks
        USE_DISTANCE_WEIGHTS = os.environ.get('NDN_SIM_USE_NEIGHBOR_DISTANCE_WEIGHTS', '0') == '1'
        if not USE_DISTANCE_WEIGHTS:
            # Fast path: use equal weights (no graph operations)
            return {neighbor_id: 1.0 for neighbor_id in neighbors}
        
        # Only do slow distance calculation if explicitly enabled
        MAX_NEIGHBORS_FOR_DISTANCE_CALC = int(os.environ.get('NDN_SIM_MAX_NEIGHBOR_DISTANCE_CALC', '10'))
        if len(neighbors) > MAX_NEIGHBORS_FOR_DISTANCE_CALC:
            # Too many neighbors - use equal weights to avoid blocking
            return {neighbor_id: 1.0 for neighbor_id in neighbors}
        
        weights = {}
        
        if router is None:
            router = self.router_ref
        if G is None:
            G = self.graph_ref
        
        if router is None or G is None:
            # Default: equal weights
            return {neighbor_id: 1.0 for neighbor_id in neighbors}
        
        try:
            import networkx as nx
            
            for neighbor_id in neighbors:
                weight = 1.0  # Base weight
                
                # Factor 1: Distance (closer neighbors are more important)
                # CRITICAL: This can be slow - only do it for small neighbor sets
                try:
                    if router.router_id in G and neighbor_id in G:
                        try:
                            path_length = nx.shortest_path_length(G, router.router_id, neighbor_id)
                            # Closer neighbors get higher weight: weight = 1.0 / (1 + distance)
                            distance_weight = 1.0 / (1.0 + path_length)
                            weight *= (0.5 + 0.5 * distance_weight)  # Scale to [0.5, 1.0]
                        except (nx.NetworkXNoPath, KeyError):
                            # No path found, use default weight
                            pass
                except Exception:
                    pass
                
                # Factor 2: Traffic volume (if tracked)
                # Check if we have statistics on messages to this neighbor
                if hasattr(router, 'stats') and hasattr(router.stats, 'stats'):
                    # Could track per-neighbor message counts here
                    # For now, use base weight
                    pass
                
                # Factor 3: Neighbor status (up/down)
                if hasattr(router, 'neighbor_status'):
                    neighbor_status = router.neighbor_status.get(neighbor_id, 'up')
                    if neighbor_status == 'down':
                        weight *= 0.1  # Down neighbors get very low weight
                
                weights[neighbor_id] = max(0.1, min(1.0, weight))  # Clamp to [0.1, 1.0]
        except Exception as e:
            logger.debug(f"ContentStore {self.router_id}: Error calculating neighbor importance: {e}")
            # Fallback: equal weights
            weights = {neighbor_id: 1.0 for neighbor_id in neighbors}
        
        return weights
            
    def store_content(self, name: str, content: Any, size: int, current_time: float, router=None, G=None) -> bool:
        """
        Store content in the cache using the selected caching policy
        
        Args:
            name: Content name
            content: Content object
            size: Content size
            current_time: Current time
            router: Router reference (for DQN state, optional)
            G: Network graph (for DQN state, optional)
        """
        # Task 2.4: Update router and graph references if provided
        if router is not None:
            self.router_ref = router
        if G is not None:
            self.graph_ref = G
        
        # Select caching policy based on mode
        if self.mode == "dqn_cache" and self.dqn_agent is not None:
            return self.store_content_with_dqn(name, content, size, current_time)
        elif self.mode in {"opt", "lfo", "fei_wang"} and self.baseline_cache is not None:
            return self.store_content_with_baseline(name, content, size, current_time, router, G)
        else:
            # Default basic caching
            return self.store_content_basic(name, content, size, current_time)
    
    def store_content_basic(self, name: str, content: Any, size: int, current_time: float) -> bool:
        """Basic caching policy"""
        with self.store_lock:
            # Task 1.4: Add detailed logging for cache insertion debugging
            logger.debug(
                f"ContentStore {self.router_id}: store_content_basic called for {name} "
                f"(size={size}, remaining={self.remaining_capacity}, "
                f"total={self.total_capacity}, store_size={len(self.store)})"
            )
            
            # Check if content already exists
            if name in self.store:
                # Update access time
                self.last_access_time[name] = current_time
                self.access_count[name] = self.access_count.get(name, 0) + 1
                if self.replacement_policy in {"fifo", "lifo"}:
                    try:
                        self.store_order.remove(name)
                    except ValueError:
                        pass
                    self.store_order.append(name)
                logger.debug(f"ContentStore {self.router_id}: Content {name} already cached, updated access")
                return True
                
            # Check if content is too large
            if size > self.total_capacity:
                logger.warning(
                    f"ContentStore {self.router_id}: Content {name} size {size} "
                    f"exceeds total capacity {self.total_capacity}"
                )
                return False

            if self.replacement_policy == "2-lru":
                if not self._twolru_should_admit(name):
                    logger.debug(
                        f"ContentStore {self.router_id}: 2-LRU filter denied {name}; awaiting second request"
                    )
                    return False
                
            # Check if we need to evict
            eviction_attempts = 0
            while self.remaining_capacity < size:
                eviction_attempts += 1
                if eviction_attempts > 10:  # Prevent infinite loops
                    logger.error(
                        f"ContentStore {self.router_id}: Too many eviction attempts for {name}, aborting"
                    )
                    return False
                # Task 2.1: Pass current_time to eviction for combined algorithm
                eviction_result = self.evict_by_policy(size - self.remaining_capacity, current_time=current_time)
                if not eviction_result:
                    logger.warning(
                        f"ContentStore {self.router_id}: Failed to evict enough space for {name} "
                        f"(needed={size}, remaining={self.remaining_capacity}, "
                        f"store_size={len(self.store)})"
                    )
                    return False
                    
            # Store the content
            if hasattr(content, 'clone'):
                content = content.clone()
            self.store[name] = content
            self.size_map[name] = size
            self.remaining_capacity -= size
            self.last_access_time[name] = current_time
            self.access_count[name] = self.access_count.get(name, 0) + 1
            self.bloom_filter.add(name)
            self.insertions += 1
            if self.replacement_policy in {"fifo", "lifo"}:
                self.store_order.append(name)
            elif self.replacement_policy in {"lru", "2-lru"}:
                try:
                    self.store_order.remove(name)
                except ValueError:
                    pass
                self.store_order.append(name)
            if self.replacement_policy == "2-lru":
                self.twolru_filter.pop(name, None)
            
            # Update clusters
            self.update_clusters(name, current_time)
            
            # Propagate Bloom filter to neighbors periodically
            self._maybe_propagate_bloom_filter()
            
            logger.debug(f"Stored content {name} of size {size}")
            return True
    
    def store_content_with_baseline(self, name: str, content: Any, size: int, current_time: float, router=None, G=None) -> bool:
        """Store content using baseline caching policy (OPT, LFO, or Fei Wang)"""
        if self.baseline_cache is None:
            logger.warning(f"ContentStore {self.router_id}: Baseline cache not initialized, falling back to basic")
            return self.store_content_basic(name, content, size, current_time)
        
        # Record request for baseline algorithms that track request history
        if hasattr(self.baseline_cache, 'record_request'):
            self.baseline_cache.record_request(name)
        
        # Check if baseline algorithm says we should cache
        should_cache = self.baseline_cache.should_cache(name, size, router, G)
        
        if not should_cache:
            logger.debug(f"ContentStore {self.router_id}: Baseline algorithm decided not to cache {name}")
            return False
        
        # Use basic storage logic but with baseline's decision
        with self.store_lock:
            # Check if content already exists
            if name in self.store:
                self.last_access_time[name] = current_time
                self.access_count[name] = self.access_count.get(name, 0) + 1
                return True
            
            # Check if content is too large
            if size > self.total_capacity:
                return False
            
            # Check if we have space or need to evict
            if self.remaining_capacity < size:
                # Need to evict - use baseline's eviction logic if available
                # Otherwise fall back to basic eviction
                required_space = size - self.remaining_capacity
                eviction_result = self.evict_by_policy(required_space, current_time)
                if not eviction_result:
                    logger.warning(f"ContentStore {self.router_id}: Failed to evict space for {name}")
                    return False
            
            # Store the content
            if hasattr(content, 'clone'):
                content = content.clone()
            self.store[name] = content
            self.size_map[name] = size
            self.remaining_capacity -= size
            self.last_access_time[name] = current_time
            self.access_count[name] = self.access_count.get(name, 0) + 1
            self.bloom_filter.add(name)
            self.insertions += 1
            
            # Update baseline cache state
            if hasattr(self.baseline_cache, 'cache_content'):
                self.baseline_cache.cache_content(name, content, size)
            
            # Update clusters
            self.update_clusters(name, current_time)
            
            # Propagate Bloom filter to neighbors periodically
            self._maybe_propagate_bloom_filter()
            
            logger.debug(f"Stored content {name} using baseline policy")
            return True
    
    def store_content_with_dqn(self, name: str, content: Any, size: int, current_time: float) -> bool:
        """Store content using DQN policy for caching decisions"""
        # OPTIMIZATION: Option to disable DQN entirely for testing if it's causing blocking
        # Set NDN_SIM_DISABLE_DQN_FOR_TESTING=1 to bypass DQN and use basic caching
        if os.environ.get('NDN_SIM_DISABLE_DQN_FOR_TESTING', '0') == '1':
            return self.store_content_basic(name, content, size, current_time)
        
        # Error handling: Validate DQN agent is available
        if self.dqn_agent is None:
            logger.warning(f"ContentStore {self.router_id}: DQN agent not initialized, falling back to basic caching")
            return self.store_content_basic(name, content, size, current_time)
        
        # CRITICAL FIX: Check basic conditions first, then do state computation and DQN inference outside lock
        # This prevents blocking all workers on a single ContentStore's lock during slow graph operations and GPU operations
        with self.store_lock:
            # Check if content already exists
            if name in self.store:
                # Update access time and count
                self.last_access_time[name] = current_time
                self.access_count[name] = self.access_count.get(name, 0) + 1
                return True
                
            # Check if content is too large
            if size > self.total_capacity:
                return False
                
        # CRITICAL FIX: Compute state OUTSIDE the lock to prevent blocking
        # State computation involves graph algorithms (shortest_path_length) which can be slow
        # The lock is only needed for reading cache state, which we do in a separate fast call
        try:
            state = self.get_state_for_dqn(name, size, router=self.router_ref, G=self.graph_ref, current_time=current_time, lock_already_held=False)
        except Exception as e:
            logger.warning(f"ContentStore {self.router_id}: Error getting DQN state: {e}, defaulting to cache")
            state = None
        
        # CRITICAL FIX: Do DQN inference OUTSIDE the lock to prevent blocking
        # GPU operations (MPS/CUDA) can be slow or blocking, so we release the lock first
        # This allows other workers to access the ContentStore while DQN inference happens
        if state is not None and len(state) >= 5:
            cache_entry = self._decision_cache.get(name)
            if cache_entry and (current_time - cache_entry[1]) <= self.decision_cache_ttl:
                action = cache_entry[0]
            else:
                try:
                    action = self._select_action_with_batching(state)
                except Exception as e:
                    logger.warning(f"ContentStore {self.router_id}: Error in DQN inference: {e}, defaulting to cache")
                    action = 1
                self._decision_cache[name] = (action, current_time)
                if len(self._decision_cache) > 2000:
                    self._decision_cache.pop(next(iter(self._decision_cache)))
        else:
            # Invalid state or error - default to caching
            action = 1
        
        # CRITICAL FIX 1: Safety check - force cache if there's available capacity
        # This prevents DQN from rejecting valid caching opportunities
        # DQN starts with epsilon=1.0 (100% random), so it may reject caching even when space is available
        # We override the decision but still record the original for learning
        cache_is_empty = False
        has_full_capacity = False
        with self.store_lock:
            cache_is_empty = len(self.store) == 0
            has_full_capacity = self.remaining_capacity >= size
        
        original_action = action  # Save original DQN decision for experience recording
        action_overridden = False
        # IMPROVEMENT: Force cache when there's available capacity (not just when empty)
        # This ensures we don't waste caching opportunities when space is clearly available
        if has_full_capacity and action == 0:
            # Override DQN decision - always cache when there's available capacity
            # This ensures cache gets populated and prevents rejecting valid opportunities
            action = 1
            action_overridden = True
            logger.debug(
                f"ContentStore {self.router_id}: Overriding DQN decision (action=0→1) "
                f"for available capacity: {name} (remaining={self.remaining_capacity}, "
                f"size={size}, original_action={original_action})"
            )
        
        # Re-acquire lock for cache operations (must be thread-safe)
        with self.store_lock:
            # RACE CONDITION CHECK: Content might have been added by another worker while we did DQN inference
            if name in self.store:
                # Another worker cached it - just update access time
                self.last_access_time[name] = current_time
                self.access_count[name] = self.access_count.get(name, 0) + 1
                return True
            
            # Process the action
            if action == 1:  # Cache the content
                # Check if we need to evict
                eviction_attempts = 0
                while self.remaining_capacity < size:
                    space_needed = size - self.remaining_capacity
                    eviction_attempts += 1
                    if eviction_attempts > 5:  # Prevent infinite loops
                        # IMPROVEMENT: Aggressive eviction fallback - clear entire cache if content fits
                        # This ensures we can always cache if the content fits in total capacity
                        if size <= self.total_capacity:
                            # Last resort: clear entire cache if content fits
                            logger.warning(
                                f"ContentStore {self.router_id}: Eviction failed {eviction_attempts} times, "
                                f"clearing entire cache to make room for {name} (size={size}, capacity={self.total_capacity})"
                            )
                            self.store.clear()
                            self.size_map.clear()
                            self.store_order.clear()
                            self.remaining_capacity = self.total_capacity
                            # Clear DQN tracking
                            self.dqn_cached_contents.clear()
                            self.dqn_decision_states.clear()
                            break  # Exit eviction loop - we now have space
                        else:
                            # Content is too large - can't cache
                            logger.warning(
                                f"ContentStore {self.router_id}: Cannot cache {name} - "
                                f"size {size} exceeds total capacity {self.total_capacity}"
                            )
                            if self.dqn_agent is not None:
                                # Learn from this experience - negative reward for failed caching
                                try:
                                    latency_fail, bandwidth_fail, hop_fail = self._get_network_reward_signals(size)
                                    reward = self.dqn_agent.calculate_reward(
                                        is_caching_decision=False,
                                        was_cached=False,
                                        latency_saved=latency_fail,
                                        bandwidth_saved=bandwidth_fail,
                                        hop_saved=hop_fail
                                    )
                                    next_state = self.get_state_for_dqn(name, size, router=self.router_ref, G=self.graph_ref, current_time=current_time, lock_already_held=True)
                                    if next_state is not None and len(next_state) >= 5:
                                        self.dqn_agent.remember(state, action, reward, next_state, False)
                                        self._maybe_train_dqn()
                                except Exception as e:
                                    logger.debug(f"ContentStore {self.router_id}: Error recording failed caching experience: {e}")
                            return False
                    else:
                        # IMPROVEMENT: Add eviction debugging
                        logger.debug(
                            f"ContentStore {self.router_id}: Eviction attempt {eviction_attempts}: "
                            f"needed={space_needed}, remaining={self.remaining_capacity}, "
                            f"store_size={len(self.store)}, capacity={self.total_capacity}"
                        )
                        
                        # Try policy-based eviction first
                        eviction_result = self.evict_by_policy(size - self.remaining_capacity, current_time=current_time)
                        
                        if not eviction_result:
                            # If policy-based eviction failed, try combined eviction as fallback
                            logger.debug(
                                f"ContentStore {self.router_id}: Policy eviction failed, "
                                f"trying combined eviction algorithm"
                            )
                            eviction_result = self.combined_eviction_algorithm(size - self.remaining_capacity, weight=0.5, current_time=current_time)
                        
                        if not eviction_result:
                            # If all eviction methods failed, try one more time with more aggressive settings
                            logger.debug(
                                f"ContentStore {self.router_id}: Combined eviction failed, "
                                f"trying aggressive LRU eviction"
                            )
                            # Try aggressive LRU - evict until we have enough space
                            space_freed = 0
                            lru_items = sorted(
                                self.store.keys(),
                                key=lambda x: self.last_access_time.get(x, 0)
                            )
                            for item in lru_items:
                                if space_freed >= space_needed:
                                    break
                                item_size = self.size_map.get(item, 0)
                                space_freed += item_size
                                self._remove_content_no_lock(item)
                            
                            if space_freed < space_needed:
                                # Still not enough - will try again or give up
                                logger.debug(
                                    f"ContentStore {self.router_id}: Aggressive LRU freed {space_freed}, "
                                    f"still need {space_needed - space_freed} more"
                                )
                                # Continue loop to try again or hit the >5 attempts fallback
                                continue
                        else:
                            # Eviction succeeded - log success
                            logger.debug(
                                f"ContentStore {self.router_id}: Eviction succeeded after {eviction_attempts} attempts, "
                                f"remaining_capacity={self.remaining_capacity}"
                            )
                            break  # Exit eviction loop
                
                # Store the content
                if hasattr(content, 'clone'):
                    content = content.clone()
                self.store[name] = content
                self.size_map[name] = size
                self.remaining_capacity -= size
                self.last_access_time[name] = current_time
                self.access_count[name] = self.access_count.get(name, 0) + 1
                self.bloom_filter.add(name)
                self.insertions += 1
                try:
                    self.store_order.remove(name)
                except ValueError:
                    pass
                self.store_order.append(name)
                
                # Update clusters
                self.update_clusters(name, current_time)
                
                # Track that this content was cached by DQN decision
                self.dqn_cached_contents.add(name)
                
                # Store the original state when caching decision was made
                # This allows proper credit assignment when cache hit occurs
                if state is not None and len(state) >= 5:
                    self.dqn_decision_states[name] = state.copy()
                
                # Propagate Bloom filter to neighbors periodically
                self._maybe_propagate_bloom_filter()
                
                if self.dqn_agent is not None:
                    # IMPROVEMENT: Calculate reward for caching decision (small reward for popular content)
                    # Reward will also come later when cache hit occurs (via notify_cache_hit)
                    # CRITICAL: Use original_action if we overrode it (for proper learning)
                    # This ensures agent learns from its actual decision, not the override
                    experience_action = original_action if action_overridden else action
                    try:
                        # Get next state for experience storage
                        next_state = self.get_state_for_dqn(name, size, router=self.router_ref, G=self.graph_ref, current_time=current_time, lock_already_held=True)
                        if next_state is not None and len(next_state) >= 5:
                            # Calculate reward for caching decision (improved: rewards popular content)
                            access_count = self.access_count.get(name, 0)
                            total_accesses = sum(self.access_count.values()) if self.access_count else 1
                            access_frequency = access_count / max(1, total_accesses)
                            
                            # Get cluster score for reward calculation
                            cluster_id = self.cluster_manager.content_to_cluster.get(name, -1)
                            cluster_score = self.cluster_manager.get_cluster_score(cluster_id) if cluster_id >= 0 else 0.0
                            
                            edge_bonus = False
                            downstream_demand = 0.0
                            if hasattr(self.router_ref, 'connected_users'):
                                edge_bonus = len(getattr(self.router_ref, 'connected_users', [])) > 0
                            pit_entries = {}
                            if self.router_ref is not None and hasattr(self.router_ref, 'PIT'):
                                pit_obj = getattr(self.router_ref, 'PIT')
                                if hasattr(pit_obj, 'entries'):
                                    pit_entries = pit_obj.entries
                            downstream_demand = len(pit_entries.get(name, [])) if pit_entries else 0.0
                            
                            latency_saved, bandwidth_saved, hop_saved = self._get_network_reward_signals(size)
                            # Calculate reward using improved reward function
                            reward = self.dqn_agent.calculate_reward(
                                is_caching_decision=True,
                                content_size=size,
                                cluster_score=cluster_score,
                                access_frequency=access_frequency,
                                latency_saved=latency_saved,
                                bandwidth_saved=bandwidth_saved,
                                hop_saved=hop_saved,
                                is_edge_router=edge_bonus,
                                downstream_demand=downstream_demand
                            )
                            
                            self.dqn_agent.remember(state, experience_action, reward, next_state, False)
                            # Also trigger training periodically even when caching (to ensure training happens)
                            # Training will also happen on cache hits, but we want to ensure it happens regularly
                            self._maybe_train_dqn()
                    except Exception as e:
                        logger.debug(f"ContentStore {self.router_id}: Error recording caching experience: {e}")
                
                return True
            else:
                # Decision not to cache
                if self.dqn_agent is not None:
                    # Small negative reward for not caching
                    try:
                        latency_saved_nc, bandwidth_saved_nc, hop_saved_nc = self._get_network_reward_signals(size)
                        reward = self.dqn_agent.calculate_reward(
                            is_caching_decision=False,
                            was_cached=False,
                            latency_saved=latency_saved_nc,
                            bandwidth_saved=bandwidth_saved_nc,
                            hop_saved=hop_saved_nc
                        )
                        next_state = self.get_state_for_dqn(name, size, router=self.router_ref, G=self.graph_ref, current_time=current_time, lock_already_held=True)
                        if next_state is not None and len(next_state) >= 5:
                            self.dqn_agent.remember(state, action, reward, next_state, False)
                            self._maybe_train_dqn()
                    except Exception as e:
                        logger.debug(f"ContentStore {self.router_id}: Error recording non-caching experience: {e}")
                
                return False
    
    def _maybe_train_dqn(self):
        """
        Schedule DQN training asynchronously to avoid blocking message processing.
        Training happens in background thread pool.
        """
        if self.dqn_agent is None:
            return
        
        self.dqn_training_step += 1
        
        # Check if training should happen
        memory_size = len(self.dqn_agent.memory)
        batch_size = self.dqn_agent.batch_size
        
        # OPTIMIZED: More aggressive training schedule for faster learning
        # Train if we have at least quarter batch size OR every N steps
        should_train = (
            self.dqn_training_step % self.dqn_training_frequency == 0 or 
            memory_size >= batch_size or
            memory_size >= (batch_size // 4)  # OPTIMIZED: Train with smaller batches (32 for batch_size=128)
        )
        
        # OPTIMIZED: Lower threshold for training (batch_size // 4 = 32 for batch_size=128)
        if should_train and memory_size >= (batch_size // 4):
            # Get training manager singleton
            try:
                from router import DQNTrainingManager
                training_manager = DQNTrainingManager.get_instance()
                
                # Log training submission for verification (use print to bypass QUIET_MODE)
                memory_size = len(self.dqn_agent.memory)
                print(f"📚 DQN Training: Router {self.router_id} queued training (memory={memory_size}, batch_size={self.dqn_agent.batch_size})", flush=True)
                logger.info(f"📚 Queuing DQN training for router {self.router_id} (memory={memory_size}, batch_size={self.dqn_agent.batch_size})")
                
                queued = training_manager.queue_training_request(
                    router_id=self.router_id,
                    training_fn=lambda: self.dqn_agent.replay(),
                    priority=memory_size,
                    memory_size=memory_size,
                )
                if not queued:
                    logger.warning(f"ContentStore {self.router_id}: Training queue unavailable, running synchronously")
                    self.dqn_agent.replay()
            except Exception as e:
                # Fallback to synchronous training if manager not available
                logger.debug(f"ContentStore {self.router_id}: Training manager not available, using sync training: {e}")
                self.dqn_agent.replay()
    
    def _get_network_reward_signals(self, content_size: int) -> Tuple[float, float, float]:
        """
        Fetch network-level signals (latency, bandwidth, hops) for reward shaping.
        """
        latency_saved = 0.2
        bandwidth_saved = float(content_size)
        hop_saved = 0.0
        try:
            from metrics import get_metrics_collector
            metrics_collector = get_metrics_collector()
            latency_metrics = metrics_collector.get_latency_metrics()
            latency_saved = latency_metrics.get('mean', latency_saved)
            hop_metrics = metrics_collector.get_hop_latency_metrics()
            hop_saved = hop_metrics.get('mean', hop_saved)
        except Exception as e:
            logger.debug(f"ContentStore {self.router_id}: Unable to fetch reward metrics: {e}")
        return latency_saved, bandwidth_saved, hop_saved
    
    def notify_cache_hit(self, content_name: str, current_time: float):
        """
        Notify ContentStore that a cache hit occurred for delayed reward.
        This is called when content cached by DQN is actually used.
        
        Args:
            content_name: Name of the content that was hit
            current_time: Current simulation time
        """
        if self.dqn_agent is None:
            return
        
        with self.store_lock:
            # Check if this content was cached by DQN decision
            if content_name not in self.dqn_cached_contents:
                return  # Not cached by DQN, no delayed reward
            
            # Calculate delayed reward for cache hit
            cluster_id = self.cluster_manager.content_to_cluster.get(content_name, -1)
            cluster_score = self.cluster_manager.get_cluster_score(cluster_id) if cluster_id >= 0 else 0
            content_size = self.size_map.get(content_name, 0)
            
            # Large positive reward for actual cache hit
            access_count = self.access_count.get(content_name, 0)
            total_accesses = sum(self.access_count.values()) if self.access_count else 1
            access_frequency = access_count / max(1, total_accesses)
            
            # Phase 6.2: Calculate latency/bandwidth/hop savings for multi-objective reward
            latency_saved, bandwidth_saved, hop_saved = self._get_network_reward_signals(content_size)
            
            reward = self.dqn_agent.calculate_reward(
                is_cache_hit=True,
                content_size=content_size,
                cluster_score=cluster_score,
                access_frequency=access_frequency,
                latency_saved=latency_saved,
                bandwidth_saved=bandwidth_saved,
                hop_saved=hop_saved
            )
            
            # Get ORIGINAL state when caching decision was made (for proper credit assignment)
            try:
                # Use original state from when caching decision was made
                original_state = self.dqn_decision_states.get(content_name)
                if original_state is None:
                    # Fallback to current state if original not found
                    original_state = self.get_state_for_dqn(content_name, content_size, router=self.router_ref, G=self.graph_ref, current_time=current_time, lock_already_held=True)
                
                # Get current state for next_state
                next_state = self.get_state_for_dqn(content_name, content_size, router=self.router_ref, G=self.graph_ref, current_time=current_time, lock_already_held=True)
                
                # Use original state for proper credit assignment
                state = original_state
                
                if state is not None and len(state) >= 5:
                    # Create experience tuple with delayed reward
                    # Action was 1 (cache), and we got a hit
                    action = 1
                    done = False
                    
                    # Store experience with delayed reward
                    self.dqn_agent.remember(state, action, reward, next_state, done)
                    
                    # Train if needed
                    self._maybe_train_dqn()
                    
                    logger.debug(f"ContentStore {self.router_id}: Delayed reward {reward:.2f} for cache hit on {content_name}")
                else:
                    logger.debug(f"ContentStore {self.router_id}: Invalid state for cache hit reward, skipping")
            except Exception as e:
                logger.debug(f"ContentStore {self.router_id}: Error processing cache hit reward: {e}")
    
    def _maybe_propagate_bloom_filter(self):
        """
        Propagate Bloom filter to neighbors periodically.
        Also propagates exact cache state if FeiWang mode is enabled.
        This enables neighbor awareness in DQN state space.
        """
        if self.router_ref is None:
            return
        
        # Check if it's time to propagate
        if (self.insertions - self.last_bloom_propagation) < self.bloom_propagation_frequency:
            return
        
        self.last_bloom_propagation = self.insertions
        
        # Check if FeiWang mode (exact neighbor state) is enabled
        use_exact_neighbor = os.environ.get("NDN_SIM_USE_EXACT_NEIGHBOR_STATE", "0") == "1"
        if use_exact_neighbor:
            # FeiWang mode: Propagate exact cache state instead of Bloom filter
            self.propagate_exact_cache_state()
        else:
            # Normal mode: Propagate Bloom filter
            self.propagate_bloom_filter_to_neighbors()
    
    def propagate_bloom_filter_to_neighbors(self):
        """
        Send local Bloom filter to all neighbors for cache state awareness.
        This enables DQN feature 4 (neighbor has content via Bloom filters).
        Tracks communication overhead for metrics collection.
        
        Phase 7.1: Respects NDN_SIM_DISABLE_BLOOM flag for ablation study.
        """
        # Phase 7.1: Support for ablation study - skip propagation if disabled
        disable_bloom = os.environ.get("NDN_SIM_DISABLE_BLOOM", "0") == "1"
        if disable_bloom:
            return  # Skip Bloom filter propagation for ablation study
        
        if self.router_ref is None:
            return
        
        # Fix: Thread-safe access to neighbors
        with self.router_ref.neighbor_lock:
            neighbors = list(self.router_ref.neighbors)  # Create a copy to avoid holding lock
        
        if not neighbors:
            return
        
        # Create a copy of the Bloom filter for sending
        # For basic Bloom filter, copy the bit array
        try:
            with self.bloom_filter.lock:
                if isinstance(self.bloom_filter, NeuralBloomFilter):
                    # For Neural Bloom Filter, create a basic Bloom filter copy
                    # (neural network state is not propagated)
                    filter_copy = BloomFilter(
                        size=self.bloom_filter.size,
                        hash_count=self.bloom_filter.hash_count
                    )
                    filter_copy.bit_array = self.bloom_filter.bit_array.copy()
                else:
                    # For basic Bloom filter, create a copy
                    filter_copy = BloomFilter(
                        size=self.bloom_filter.size,
                        hash_count=self.bloom_filter.hash_count
                    )
                    filter_copy.bit_array = self.bloom_filter.bit_array.copy()
            
            # Calculate Bloom filter size in bytes for overhead tracking
            bloom_filter_bytes = (self.bloom_filter.size + 7) // 8  # Convert bits to bytes
            
            # Send to all neighbors and track communication overhead
            for neighbor_id in neighbors:
                if self.router_ref and hasattr(self.router_ref, 'send_message'):
                    try:
                        self.router_ref.send_message(
                            neighbor_id,
                            'bloom_filter_update',
                            (self.router_id, filter_copy),
                            priority=2  # Lower priority than Interest/Data packets
                        )
                        
                        # Track communication overhead (Bloom filter propagation)
                        try:
                            from metrics import get_metrics_collector
                            metrics_collector = get_metrics_collector()
                            # Track Bloom filter overhead as Interest bytes (communication overhead)
                            metrics_collector.record_interest(
                                f"bloom_filter_{self.router_id}_{neighbor_id}",
                                f"bloom_filter_update",
                                self.router_id,
                                interest_size=bloom_filter_bytes
                            )
                        except Exception as e:
                            logger.debug(f"ContentStore {self.router_id}: Error tracking Bloom filter overhead: {e}")
                        
                        logger.debug(f"ContentStore {self.router_id}: Propagated Bloom filter to neighbor {neighbor_id}")
                    except Exception as e:
                        logger.debug(f"ContentStore {self.router_id}: Failed to send Bloom filter to {neighbor_id}: {e}")
        except Exception as e:
            logger.debug(f"ContentStore {self.router_id}: Error propagating Bloom filter: {e}")
    
    def receive_bloom_filter_update(self, neighbor_id: int, bloom_filter: BloomFilter):
        """
        Receive and store neighbor's Bloom filter update.
        This updates the neighbor_filters dictionary used in DQN state space.
        
        Phase 3.3: Initializes false positive tracking for this neighbor.
        
        Args:
            neighbor_id: ID of the neighbor router
            bloom_filter: Bloom filter from the neighbor
        """
        try:
            # Store the neighbor's Bloom filter
            # Create a new filter to avoid reference issues
            stored_filter = BloomFilter(
                size=bloom_filter.size,
                hash_count=bloom_filter.hash_count
            )
            with bloom_filter.lock:
                stored_filter.bit_array = bloom_filter.bit_array.copy()
            
            self.neighbor_filters[neighbor_id] = stored_filter
            
            # Phase 3.3: Initialize false positive tracking for this neighbor
            if not hasattr(self, 'neighbor_false_positives'):
                self.neighbor_false_positives: Dict[int, Dict[str, int]] = {}  # neighbor_id -> tracking data
            if neighbor_id not in self.neighbor_false_positives:
                self.neighbor_false_positives[neighbor_id] = {'total_checks': 0, 'false_positives': 0}
            
            logger.debug(f"ContentStore {self.router_id}: Received Bloom filter update from neighbor {neighbor_id}")
        except Exception as e:
            logger.debug(f"ContentStore {self.router_id}: Error receiving Bloom filter from {neighbor_id}: {e}")
    
    def propagate_exact_cache_state(self):
        """
        FeiWang mode: Propagate exact cache contents to all neighbors.
        This implements the exact neighbor state exchange from Fei Wang et al. (ICC 2023).
        
        Unlike Bloom filters, this sends the complete list of cached content names.
        Higher communication overhead but exact state information.
        """
        use_exact_neighbor = os.environ.get("NDN_SIM_USE_EXACT_NEIGHBOR_STATE", "0") == "1"
        if not use_exact_neighbor:
            return
        
        try:
            # Get neighbors
            neighbors = []
            if self.router_ref and hasattr(self.router_ref, 'neighbors'):
                with self.router_ref.neighbor_lock:
                    neighbors = list(self.router_ref.neighbors)
            
            if not neighbors:
                return
            
            # Get exact cache contents (set of content names)
            with self.store_lock:
                cache_contents = set(self.store.keys())
            
            # Calculate exact state size in bytes (approximate: content names as strings)
            # Each content name is typically 20-50 bytes, so estimate
            estimated_bytes = sum(len(name) for name in cache_contents) if cache_contents else 0
            
            # Send to all neighbors
            for neighbor_id in neighbors:
                if self.router_ref and hasattr(self.router_ref, 'send_message'):
                    try:
                        # Send exact cache contents (set of content names)
                        self.router_ref.send_message(
                            neighbor_id,
                            'exact_cache_update',
                            (self.router_id, cache_contents.copy()),  # Send copy to avoid race conditions
                            priority=2  # Lower priority than Interest/Data packets
                        )
                        
                        # Track communication overhead (exact state propagation)
                        try:
                            from metrics import get_metrics_collector
                            metrics_collector = get_metrics_collector()
                            # Track exact state overhead as Interest bytes (communication overhead)
                            metrics_collector.record_interest(
                                f"exact_state_{self.router_id}_{neighbor_id}",
                                f"exact_cache_update",
                                self.router_id,
                                interest_size=estimated_bytes
                            )
                        except Exception as e:
                            logger.debug(f"ContentStore {self.router_id}: Error tracking exact state overhead: {e}")
                        
                        logger.debug(f"ContentStore {self.router_id}: Propagated exact cache state to neighbor {neighbor_id} ({len(cache_contents)} items)")
                    except Exception as e:
                        logger.debug(f"ContentStore {self.router_id}: Failed to send exact cache state to {neighbor_id}: {e}")
        except Exception as e:
            logger.debug(f"ContentStore {self.router_id}: Error propagating exact cache state: {e}")
    
    def receive_exact_cache_update(self, neighbor_id: int, cache_contents: set):
        """
        FeiWang mode: Receive and store exact neighbor cache contents.
        This updates the neighbor_exact_states dictionary used in DQN state space.
        
        Args:
            neighbor_id: ID of the neighbor router
            cache_contents: Set of content names cached by the neighbor (exact state)
        """
        try:
            # Store exact neighbor cache state
            if not hasattr(self, 'neighbor_exact_states'):
                self.neighbor_exact_states = {}
            
            # Store copy to avoid reference issues
            self.neighbor_exact_states[neighbor_id] = cache_contents.copy()
            
            logger.debug(f"ContentStore {self.router_id}: Received exact cache state from neighbor {neighbor_id} ({len(cache_contents)} items)")
        except Exception as e:
            logger.debug(f"ContentStore {self.router_id}: Error receiving exact cache state from {neighbor_id}: {e}")
    
    def _track_bloom_filter_false_positive(self, neighbor_id: int, content_name: str, actually_has: bool):
        """
        Phase 3.3: Track false positives for Bloom filter learning
        
        Args:
            neighbor_id: ID of neighbor
            content_name: Content name that was checked
            actually_has: True if neighbor actually has the content, False if false positive
        """
        if not hasattr(self, 'neighbor_false_positives'):
            self.neighbor_false_positives: Dict[int, Dict[str, int]] = {}
        
        if neighbor_id not in self.neighbor_false_positives:
            self.neighbor_false_positives[neighbor_id] = {'total_checks': 0, 'false_positives': 0}
        
        self.neighbor_false_positives[neighbor_id]['total_checks'] += 1
        if not actually_has:
            self.neighbor_false_positives[neighbor_id]['false_positives'] += 1
        
        # Calculate false positive rate for this neighbor
        fp_data = self.neighbor_false_positives[neighbor_id]
        if fp_data['total_checks'] > 0:
            fpr = fp_data['false_positives'] / fp_data['total_checks']
            # Adjust confidence in future checks based on FPR
            # Higher FPR = lower confidence
            if fpr > 0.1:  # If FPR > 10%, reduce weight
                logger.debug(f"ContentStore {self.router_id}: Neighbor {neighbor_id} has high FPR {fpr:.3f}, reducing confidence")
            
    def get_content(self, name: str, current_time: float) -> Optional[Any]:
        """Retrieve content from cache"""
        with self.store_lock:
            content = self.store.get(name)
            if content is not None:
                # Update access statistics
                self.last_access_time[name] = current_time
                self.access_count[name] = self.access_count.get(name, 0) + 1
                if self.replacement_policy == "lru":
                    try:
                        self.store_order.remove(name)
                    except ValueError:
                        pass
                    self.store_order.append(name)
                
                # Update cluster score
                cluster_id = self.cluster_manager.content_to_cluster.get(name)
                if cluster_id is not None:
                    self.cluster_manager.update_cluster_score(cluster_id, 1.0, current_time)
                    
                logger.debug(f"Cache hit for {name}")
                if hasattr(content, 'clone'):
                    content = content.clone()
            return content
            
    def _remove_content_no_lock(self, name: str):
        """Internal helper to remove content with lock already held."""
        if name in self.store:
            self.remaining_capacity += self.size_map.get(name, 0)
            
            del self.store[name]
            if name in self.size_map:
                del self.size_map[name]
            try:
                self.store_order.remove(name)
            except ValueError:
                pass
            # Remove from DQN tracking if it was cached by DQN
            self.dqn_cached_contents.discard(name)
            # Remove cached embedding if it exists
            self.cached_embeddings.pop(name, None)
            logger.debug(f"Removed content {name} from cache")

    def remove_content(self, name: str):
        """Remove content from cache"""
        with self.store_lock:
            self._remove_content_no_lock(name)
                
    def get_cluster_statistics(self) -> Dict:
        """Get statistics about content clusters"""
        with self.store_lock, self.embedding_lock:
            return {
                'num_clusters': len(self.cluster_manager.clusters),
                'cluster_sizes': {k: len(v) for k, v in self.cluster_manager.clusters.items()},
                'cluster_scores': dict(self.cluster_manager.cluster_scores),
                'capacity_used': self.total_capacity - self.remaining_capacity,
                'total_content': len(self.store)
            }


@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any = field(compare=False)
    timestamp: datetime = field(default_factory=datetime.now, compare=True)


class PIT:
    def __init__(self, threshold: int = 500):
        self.entries: Dict[str, List[int]] = {}
        self.times: Dict[str, float] = {}
        self.lock = threading.Lock()
        self.threshold = threshold
        
    def add_entry(self, name: str, incoming_face: int, router_time: float):
        with self.lock:
            if name not in self.entries:
                self.entries[name] = []
                self.times[name] = router_time
            if incoming_face not in self.entries[name]:
                self.entries[name].append(incoming_face)
                self.times[name] = router_time
                
    def remove_entry(self, name: str, incoming_face: int = None):
        with self.lock:
            if name in self.entries:
                if incoming_face is not None and incoming_face in self.entries[name]:
                    self.entries[name].remove(incoming_face)
                    if not self.entries[name]:
                        del self.entries[name]
                        if name in self.times:
                            del self.times[name]
                else:
                    del self.entries[name]
                    if name in self.times:
                        del self.times[name]
                    
    def get(self, name: str) -> List[int]:
        with self.lock:
            return self.entries.get(name, []).copy()
    
    def cleanup_expired(self, current_time: float, interest_lifetime: float = 4.0):
        """
        FIX #4: Remove expired PIT entries based on Interest lifetime (RFC 8569)
        
        Args:
            current_time: Current router time
            interest_lifetime: Maximum lifetime for Interest packets (default 4.0 seconds)
        """
        with self.lock:
            expired_names = []
            for name, entry_time in list(self.times.items()):
                if current_time - entry_time > interest_lifetime:
                    expired_names.append(name)
            
            for name in expired_names:
                if name in self.entries:
                    del self.entries[name]
                if name in self.times:
                    del self.times[name]
            
            if expired_names:
                logger.debug(f"PIT: Cleaned up {len(expired_names)} expired entries")
            
    def __contains__(self, name: str) -> bool:
        with self.lock:
            return name in self.entries
            # Fall back to basic caching if DQN fails
            

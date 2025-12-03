#!/usr/bin/env python3
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import threading
import os
from pathlib import Path

# Check if TensorBoard is available, but make it optional
try:
    from torch.utils.tensorboard import SummaryWriter
    USE_TENSORBOARD = True
except ImportError:
    USE_TENSORBOARD = False
    print("TensorBoard not available, will not log training metrics")

# CRITICAL FIX: Global semaphore for all GPU/MPS operations to prevent deadlocks
# MPS (Apple Silicon GPU) can deadlock when multiple threads access simultaneously
# This global semaphore serializes ALL GPU access across all DQN agents (only 1 at a time)
# Using Semaphore instead of Lock to support timeout
_GLOBAL_GPU_SEMAPHORE = threading.Semaphore(1)  # Binary semaphore = lock with timeout support

# CRITICAL FIX: Semaphore to limit concurrent DQN operations
# Prevents all workers from blocking on DQN inference simultaneously
# Only allow 4 concurrent DQN operations at a time (others will wait but not block indefinitely)
_DQN_SEMAPHORE = threading.Semaphore(int(os.environ.get('NDN_SIM_DQN_CONCURRENCY', '4')))

def get_device():
    """Get appropriate device for training"""
    # CRITICAL FIX: Force CPU for multithreaded simulations
    # MPS (Apple Silicon GPU) has known threading issues that cause deadlocks
    # when multiple threads access it simultaneously, even with locks
    # See: https://github.com/pytorch/pytorch/issues/77799
    force_cpu = os.environ.get('NDN_SIM_FORCE_CPU_FOR_DQN', '1') == '1'
    if force_cpu:
        return torch.device('cpu')
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')  # Use Metal Performance Shaders on Mac
    return torch.device('cpu')


class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 128, 64]):
        super(DQNNetwork, self).__init__()
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.2)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool, error: float):
        priority = (abs(error) + 1e-5) ** self.alpha
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(priority)
        
    def sample(self, batch_size: int) -> Tuple:
        total = len(self.memory)
        if total < batch_size:
            # If not enough samples, return all with equal weights
            indices = list(range(total))
            samples = list(self.memory)
            weights = np.ones(total)
            return samples, indices, weights
            
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        indices = np.random.choice(total, batch_size, p=probs, replace=False)
        samples = [self.memory[idx] for idx in indices]
        # Importance sampling weights
        weights = (total * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        # Increase beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        return samples, indices, weights
        
    def update_priorities(self, indices: List[int], errors: np.ndarray):
        for idx, error in zip(indices, errors):
            if idx < len(self.priorities):  # Safeguard against out of bounds
                self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha
            
    def __len__(self) -> int:
        return len(self.memory)

class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [512, 256, 128],  # OPTIMIZED: Larger network for better approximation
        lr: float = 5e-4,  # OPTIMIZED: Increased from 3e-4 for faster convergence
        gamma: float = 0.995,  # OPTIMIZED: Increased from 0.99 for longer-term credit assignment
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,  # OPTIMIZED: Higher minimum exploration (0.05 vs 0.01)
        epsilon_decay: float = 0.9995,  # OPTIMIZED: Slower decay for sustained exploration
        batch_size: int = 128,  # OPTIMIZED: Increased from 64 for more stable gradients
        memory_size: int = 10000,
        target_update_freq: int = 200,  # OPTIMIZED: Increased from 100 for more stable Q-targets
        n_step: int = 30,  # OPTIMIZED: Increased from 20 for better long-term reward propagation
        router_id: Optional[int] = None,
        experiment_id: Optional[str] = None,
        tensorboard_dir: Optional[str] = None
    ):
        self.device = get_device()
        print(f"Using device: {self.device}")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.n_step = n_step
        self.step_count = 0
        
        # Initialize networks
        self.policy_net = DQNNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize RL components
        self.memory = PrioritizedReplayBuffer(memory_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min')
        self.router_id = router_id
        self.experiment_id = experiment_id or os.environ.get('NDN_SIM_EXPERIMENT_ID')
        self.tensorboard_dir = tensorboard_dir or os.environ.get('NDN_SIM_TENSORBOARD_DIR', 'runs')
        # OPTIMIZED Reward shaping parameters (configurable via environment variables)
        self.hit_reward = float(os.environ.get('NDN_SIM_DQN_HIT_REWARD', '20.0'))  # OPTIMIZED: 15.0 -> 20.0
        self.cluster_reward_weight = float(os.environ.get('NDN_SIM_DQN_CLUSTER_WEIGHT', '1.5'))  # OPTIMIZED: 1.0 -> 1.5
        self.frequency_reward_weight = float(os.environ.get('NDN_SIM_DQN_FREQUENCY_WEIGHT', '3.0'))  # OPTIMIZED: 2.0 -> 3.0
        self.size_penalty_scale = float(os.environ.get('NDN_SIM_DQN_SIZE_PENALTY', '0.08'))  # OPTIMIZED: 0.1 -> 0.08
        self.latency_reward_weight = float(os.environ.get('NDN_SIM_DQN_LATENCY_WEIGHT', '0.15'))  # OPTIMIZED: 0.1 -> 0.15
        self.bandwidth_reward_weight = float(os.environ.get('NDN_SIM_DQN_BANDWIDTH_WEIGHT', '0.00015'))  # OPTIMIZED: 0.0001 -> 0.00015
        self.latency_decision_weight = float(os.environ.get('NDN_SIM_DQN_DECISION_LATENCY_WEIGHT', '0.05'))
        self.bandwidth_decision_weight = float(os.environ.get('NDN_SIM_DQN_DECISION_BANDWIDTH_WEIGHT', '0.00005'))
        self.edge_bonus = float(os.environ.get('NDN_SIM_DQN_EDGE_BONUS', '0.5'))  # OPTIMIZED: 0.3 -> 0.5
        self.downstream_demand_weight = float(os.environ.get('NDN_SIM_DQN_DEMAND_WEIGHT', '0.15'))  # OPTIMIZED: 0.1 -> 0.15
        self.downstream_demand_cap = float(os.environ.get('NDN_SIM_DQN_DEMAND_CAP', '0.5'))  # OPTIMIZED: 0.4 -> 0.5
        self.cache_miss_penalty = float(os.environ.get('NDN_SIM_DQN_MISS_PENALTY', '2.5'))  # OPTIMIZED: 2.0 -> 2.5
        self.skip_pop_penalty = float(os.environ.get('NDN_SIM_DQN_SKIP_POP_PENALTY', '0.8'))  # OPTIMIZED: 0.5 -> 0.8
        self.unpopular_skip_penalty = float(os.environ.get('NDN_SIM_DQN_SKIP_UNPOPULAR_PENALTY', '0.15'))  # OPTIMIZED: 0.1 -> 0.15
        self.hop_reward_weight = float(os.environ.get('NDN_SIM_DQN_HOP_REWARD_WEIGHT', '0.5'))
        self.hop_decision_weight = float(os.environ.get('NDN_SIM_DQN_DECISION_HOP_WEIGHT', '0.1'))
        self.hop_penalty_weight = float(os.environ.get('NDN_SIM_DQN_HOP_PENALTY_WEIGHT', '0.5'))
        self.latency_scale = float(os.environ.get('NDN_SIM_DQN_LATENCY_SCALE', '1.0'))
        self.bandwidth_scale = float(os.environ.get('NDN_SIM_DQN_BANDWIDTH_SCALE', '1048576'))  # Default 1 MB
        self.hop_scale = float(os.environ.get('NDN_SIM_DQN_HOP_SCALE', '5.0'))
        self._apply_reward_config()
        
        # Initialize TensorBoard if available
        if USE_TENSORBOARD:
            log_dir = None
            try:
                base_dir = Path(self.tensorboard_dir) if self.tensorboard_dir else None
                if base_dir is not None:
                    if self.experiment_id:
                        base_dir = base_dir / self.experiment_id
                    if self.router_id is not None:
                        base_dir = base_dir / f"router_{self.router_id}"
                    log_dir = str(base_dir)
                self.writer = SummaryWriter(log_dir=log_dir)
                if self.router_id is not None:
                    self.writer.add_text("metadata/router_id", str(self.router_id), 0)
                if self.experiment_id:
                    self.writer.add_text("metadata/experiment_id", self.experiment_id, 0)
            except Exception:
                self.writer = SummaryWriter()
        else:
            self.writer = None
            
        self.training_stats = {'losses': [], 'rewards': [], 'cache_hits': 0, 'cache_misses': 0}
        self.best_reward = -float('inf')
        self.best_hit_rate = -float('inf')
        
        # Learning curve tracking: per-round metrics
        self.learning_curve: Dict[int, Dict] = {}  # {round: {hit_rate, loss, reward, epsilon, cache_decisions}}
        self.no_improvement_count = 0
        
        # Checkpoint management for research
        self.checkpoint_dir = None
        self.checkpoint_frequency = 10  # Save every N rounds
        self.keep_checkpoints = 5  # Keep last N checkpoints
        self.best_model_path = None
        self.last_checkpoint_round = 0

    def _apply_reward_config(self):
        """Optionally load reward-related overrides from a JSON config file."""
        config_path = os.environ.get('NDN_SIM_DQN_REWARD_CONFIG')
        if not config_path:
            return
        try:
            with open(config_path, 'r') as fp:
                overrides = json.load(fp)
        except Exception as exc:
            print(f"[DQN] Warning: Failed to load reward config {config_path}: {exc}")
            return
        
        reward_keys = {
            'hit_reward',
            'cluster_reward_weight',
            'frequency_reward_weight',
            'size_penalty_scale',
            'latency_reward_weight',
            'bandwidth_reward_weight',
            'latency_decision_weight',
            'bandwidth_decision_weight',
            'edge_bonus',
            'downstream_demand_weight',
            'downstream_demand_cap',
            'cache_miss_penalty',
            'skip_pop_penalty',
            'unpopular_skip_penalty',
            'hop_reward_weight',
            'hop_decision_weight',
            'hop_penalty_weight',
            'latency_scale',
            'bandwidth_scale',
            'hop_scale'
        }
        for key, value in overrides.items():
            if key in reward_keys:
                try:
                    setattr(self, key, float(value))
                except (TypeError, ValueError):
                    setattr(self, key, value)
        
    def get_state_features(self, state: np.ndarray) -> np.ndarray:
        """Extract and normalize relevant features from the state"""
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        # Normalize using robust statistics
        state = (state - np.mean(state, axis=1, keepdims=True)) / (np.std(state, axis=1, keepdims=True) + 1e-8)
        return state.squeeze()
        
    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy with adaptive exploration"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
            
        # CRITICAL FIX: Use semaphore to limit concurrent operations, then global lock for GPU access
        # This prevents all 37 workers from blocking simultaneously on DQN inference
        # Semaphore allows only 4 concurrent operations, others wait without blocking the entire system
        semaphore_acquired = False
        try:
            # Try to acquire semaphore with timeout to prevent indefinite blocking
            # If we can't get access within 0.2 seconds, fall back to random action
            # Reduced timeout to prevent workers from blocking too long
            semaphore_acquired = _DQN_SEMAPHORE.acquire(timeout=0.2)
            if not semaphore_acquired:
                # Semaphore timeout - too many concurrent operations, use random action
                return random.randrange(self.action_dim)
            
            # CRITICAL FIX: Use GLOBAL semaphore to serialize ALL GPU access across ALL agents
            # MPS (Apple Silicon GPU) has internal global locks that deadlock with concurrent access
            # This global semaphore prevents all 37+ agents from accessing MPS simultaneously
            # Use timeout (0.2s) to prevent indefinite blocking - reduced from 0.5s
            gpu_lock_acquired = _GLOBAL_GPU_SEMAPHORE.acquire(timeout=0.2)
            if not gpu_lock_acquired:
                # GPU lock timeout - too much contention, use random action
                return random.randrange(self.action_dim)
            
            try:
                with torch.no_grad():
                    state = self.get_state_features(state)
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    # Handle batch norm during evaluation
                    self.policy_net.eval()
                    q_values = self.policy_net(state_tensor)
                    self.policy_net.train()
                    return q_values.argmax().item()
            finally:
                _GLOBAL_GPU_SEMAPHORE.release()
        except Exception as e:
            # If DQN inference fails for any reason, fall back to random action
            # This prevents workers from getting stuck on DQN errors
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"DQN inference error: {e}, using random action")
            return random.randrange(self.action_dim)
        finally:
            # Always release semaphore if we acquired it
            if semaphore_acquired:
                _DQN_SEMAPHORE.release()
            
    def select_action_batch(self, states: List[np.ndarray]) -> List[int]:
        if not states:
            return []
        actions: List[Optional[int]] = [None] * len(states)
        greedy_indices: List[int] = []
        for idx, state in enumerate(states):
            if random.random() < self.epsilon:
                actions[idx] = random.randrange(self.action_dim)
            else:
                greedy_indices.append(idx)
        if not greedy_indices:
            return [int(action) if action is not None else 0 for action in actions]
        semaphore_acquired = False
        gpu_lock_acquired = False
        try:
            semaphore_acquired = _DQN_SEMAPHORE.acquire(timeout=0.2)
            if not semaphore_acquired:
                for idx in greedy_indices:
                    actions[idx] = random.randrange(self.action_dim)
                return [int(action) if action is not None else 0 for action in actions]
            gpu_lock_acquired = _GLOBAL_GPU_SEMAPHORE.acquire(timeout=0.2)
            if not gpu_lock_acquired:
                for idx in greedy_indices:
                    actions[idx] = random.randrange(self.action_dim)
                return [int(action) if action is not None else 0 for action in actions]
            try:
                with torch.no_grad():
                    batch_states = [self.get_state_features(states[i]) for i in greedy_indices]
                    state_tensor = torch.FloatTensor(batch_states).to(self.device)
                    self.policy_net.eval()
                    q_values = self.policy_net(state_tensor)
                    self.policy_net.train()
                    batch_actions = q_values.argmax(dim=1).tolist()
                    for idx, action in zip(greedy_indices, batch_actions):
                        actions[idx] = action
            finally:
                _GLOBAL_GPU_SEMAPHORE.release()
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"DQN batch inference error: {e}, falling back to random actions")
            for idx in greedy_indices:
                actions[idx] = random.randrange(self.action_dim)
        finally:
            if semaphore_acquired:
                _DQN_SEMAPHORE.release()
        return [int(action) if action is not None else 0 for action in actions]
    def update_epsilon(self, reward: float):
        """Adaptively update epsilon based on performance"""
        if reward > self.best_reward:
            self.best_reward = reward
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        # Slow down epsilon decay if performance is improving
        decay_rate = self.epsilon_decay if self.no_improvement_count < 5 else self.epsilon_decay * 0.95
        self.epsilon = max(self.epsilon_end, self.epsilon * decay_rate)
        
    def calculate_reward(self, is_cache_hit: bool = False, is_caching_decision: bool = False, 
                         content_size: int = 0, cluster_score: float = 0.0, was_cached: bool = False,
                         access_frequency: float = 0.0, latency_saved: float = 0.0, 
                         bandwidth_saved: float = 0.0, hop_saved: float = 0.0,
                         is_edge_router: bool = False, downstream_demand: float = 0.0) -> float:
        """
        Phase 6: Multi-objective reward function
        
        Calculate reward based on caching decision or cache hit.
        Optimizes for: hit rate, latency reduction, bandwidth savings.
        
        Args:
            is_cache_hit: True if this is a delayed reward for an actual cache hit
            is_caching_decision: True if this is an immediate reward for caching decision
            content_size: Size of the content (for size penalty)
            cluster_score: Popularity score of content's cluster
            was_cached: True if content was previously cached (for cache miss penalty)
            access_frequency: Normalized access frequency (0-1)
            latency_saved: Latency saved in seconds (for multi-objective optimization)
            bandwidth_saved: Bandwidth saved in bytes (for multi-objective optimization)
            hop_saved: Hops saved compared to fetching from producer (for topology-aware rewards)
        
        Returns:
            Reward value (scaled appropriately)
        """
        latency_scale = max(self.latency_scale, 1e-6)
        bandwidth_scale = max(self.bandwidth_scale, 1e-6)
        hop_scale = max(self.hop_scale, 1e-6)

        if is_cache_hit:
            # Phase 6: Multi-objective reward
            # Delayed reward: Large positive reward for actual cache hit
            base_reward = self.hit_reward  # Hit rate objective
            cluster_bonus = self.cluster_reward_weight * cluster_score
            frequency_bonus = self.frequency_reward_weight * access_frequency
            size_penalty = -self.size_penalty_scale * (content_size / 100)
            
            # Phase 6.1: Multi-objective components (normalized)
            latency_component = (latency_saved / latency_scale) * self.latency_reward_weight
            bandwidth_component = (bandwidth_saved / bandwidth_scale) * self.bandwidth_reward_weight
            hop_component = (hop_saved / hop_scale) * self.hop_reward_weight
            
            return base_reward + cluster_bonus + frequency_bonus + size_penalty + latency_component + bandwidth_component + hop_component
        elif is_caching_decision:
            base_reward = 0.0
            if access_frequency > 0.3 or cluster_score > 0.5:
                base_reward += 0.5
            if is_edge_router:
                base_reward += self.edge_bonus
            if downstream_demand > 0:
                base_reward += min(self.downstream_demand_cap, self.downstream_demand_weight * downstream_demand)
            base_reward += (latency_saved / latency_scale) * self.latency_decision_weight
            base_reward += (bandwidth_saved / bandwidth_scale) * self.bandwidth_decision_weight
            base_reward += (hop_saved / hop_scale) * self.hop_decision_weight
            return base_reward
        elif was_cached:
            # Cache miss when we cached it: Medium negative
            # Penalizes caching content that doesn't get hit
            hop_penalty = (hop_saved / hop_scale) * self.hop_penalty_weight
            return -(self.cache_miss_penalty + hop_penalty)  # Increased penalty from -1.0
        else:
            # Decision not to cache: Small negative (or zero if content is unpopular)
            # Encourages exploration but not too strongly
            # If content is popular, penalize not caching more
            if cluster_score > 0.5 or access_frequency > 0.3:
                return -(self.skip_pop_penalty + (hop_saved / hop_scale) * self.hop_penalty_weight)  # Penalize not caching popular content
            return -(self.unpopular_skip_penalty + (hop_saved / hop_scale) * 0.5 * self.hop_penalty_weight)  # Small penalty for not caching unpopular content
        
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """Store experience in replay buffer with priority"""
        # Initialize error with high priority in case we can't compute it
        error = 1.0
        
        # CRITICAL FIX: Normalize state features BEFORE lock to save time
        state = self.get_state_features(state)
        next_state = self.get_state_features(next_state)
        
        # CRITICAL FIX: Acquire semaphore to protect GPU/model access
        # Use timeout to prevent blocking - if we can't get lock, use default error
        gpu_lock_acquired = _GLOBAL_GPU_SEMAPHORE.acquire(timeout=0.1)
        
        if gpu_lock_acquired:
            try:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                    
                    # Handle batch norm during evaluation
                    self.policy_net.eval()
                    current_q = self.policy_net(state_tensor)[0][action]
                    self.policy_net.train()
                    
                    # Use target network for next state Q-values
                    self.target_net.eval()
                    next_q = self.target_net(next_state_tensor).max(1)[0]
                    self.target_net.train()
                    
                    expected_q = reward + (1 - done) * self.gamma * next_q
                    error = abs(current_q - expected_q).item()
            except Exception as e:
                # Fallback to default error
                pass
            finally:
                _GLOBAL_GPU_SEMAPHORE.release()
        
        # Add to replay buffer (this is thread-safe enough as it's just a deque append)
        self.memory.push(state, action, reward, next_state, done, error)
        
    def replay(self) -> float:
        """Train the network using prioritized experience replay"""
        # CRITICAL FIX: Allow training with smaller batches (at least half batch_size)
        # This ensures training happens more frequently, especially early in simulation
        min_batch_size = max(1, self.batch_size // 2)  # At least 32 experiences
        if len(self.memory) < min_batch_size:
            return 0.0
            
        # CRITICAL FIX: Acquire semaphore to protect GPU/model access
        # Use timeout to prevent blocking - if we can't get lock, skip training this time
        # Increased timeout to 10.0s since this runs in background thread and we really want to train
        gpu_lock_acquired = _GLOBAL_GPU_SEMAPHORE.acquire(timeout=10.0)
        if not gpu_lock_acquired:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"DQN training skipped due to lock timeout (10.0s)")
            return 0.0
            
        try:
            # Sample batch with priorities
            batch, indices, weights = self.memory.sample(self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # Convert to tensors
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            weights = torch.FloatTensor(weights).to(self.device)
            
            # Compute current Q values
            current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
            
            # Compute target Q values with N-Step Double DQN
            with torch.no_grad():
                if self.n_step > 1:
                    # N-step approximation: scale rewards and use longer horizon
                    effective_gamma = self.gamma ** self.n_step  # Discount over N steps
                    next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
                    next_q_values = self.target_net(next_states).gather(1, next_actions)
                    # Scale rewards to account for N-step horizon
                    scaled_rewards = rewards.unsqueeze(1) * (1 + (self.n_step - 1) * 0.1)
                    expected_q_values = scaled_rewards + (1 - dones.unsqueeze(1)) * effective_gamma * next_q_values
                else:
                    # Standard 1-step
                    next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
                    next_q_values = self.target_net(next_states).gather(1, next_actions)
                    expected_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
            
            # Compute loss with importance sampling weights
            loss = (weights.unsqueeze(1) * F.smooth_l1_loss(
                current_q_values,
                expected_q_values,
                reduction='none'
            )).mean()
            
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()
            
            # Update target network
            self.step_count += 1
            if self.step_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Update priorities in replay buffer
            errors = abs(current_q_values - expected_q_values).detach().cpu().numpy()
            self.memory.update_priorities(indices, errors.squeeze())
            
            # Update epsilon and learning rate
            self.update_epsilon(rewards.mean().item())
            self.scheduler.step(loss)
            
            # Log metrics
            grad_norm_value = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
            if self.writer is not None:
                self.writer.add_scalar('Loss/train', loss.item(), self.step_count)
                self.writer.add_scalar('Reward/train', rewards.mean().item(), self.step_count)
                self.writer.add_scalar('Epsilon', self.epsilon, self.step_count)
                self.writer.add_scalar('Grad/norm', grad_norm_value, self.step_count)
                self.writer.add_scalar('Q/current_mean', current_q_values.mean().item(), self.step_count)
                self.writer.add_scalar('Q/target_mean', expected_q_values.mean().item(), self.step_count)
                self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], self.step_count)
            
            # Track statistics
            loss_value = loss.item()
            self.training_stats['losses'].append(loss_value)
            self.training_stats['rewards'].append(rewards.mean().item())
            return loss_value
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"DQN training error: {e}")
            return 0.0
        finally:
            _GLOBAL_GPU_SEMAPHORE.release()
        
    def get_statistics(self) -> Dict:
        """Return agent's performance statistics"""
        return {
            'avg_loss': np.mean(self.training_stats['losses']) if self.training_stats['losses'] else 0,
            'avg_reward': np.mean(self.training_stats['rewards']) if self.training_stats['rewards'] else 0,
            'cache_hit_rate': (
                self.training_stats['cache_hits'] /
                (self.training_stats['cache_hits'] + self.training_stats['cache_misses'] + 1e-8)
            ),
            'epsilon': self.epsilon,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def record_round_metrics(self, round_num: int, hit_rate: float, cache_decisions: int = 0):
        """
        Record metrics for a specific round (for learning curve analysis)
        
        Args:
            round_num: Round number
            hit_rate: Cache hit rate for this round
            cache_decisions: Number of cache decisions made this round
        """
        avg_loss = np.mean(self.training_stats['losses'][-100:]) if self.training_stats['losses'] else 0.0
        avg_reward = np.mean(self.training_stats['rewards'][-100:]) if self.training_stats['rewards'] else 0.0
        
        self.learning_curve[round_num] = {
            'hit_rate': hit_rate,
            'loss': avg_loss,
            'reward': avg_reward,
            'epsilon': self.epsilon,
            'cache_decisions': cache_decisions,
            'training_steps': self.step_count
        }
        if self.writer is not None:
            self.writer.add_scalar('Round/HitRate', hit_rate, round_num)
            self.writer.add_scalar('Round/AvgLoss', avg_loss, round_num)
            self.writer.add_scalar('Round/AvgReward', avg_reward, round_num)
            self.writer.add_scalar('Round/Epsilon', self.epsilon, round_num)
            self.writer.add_scalar('Round/CacheDecisions', cache_decisions, round_num)
            self.writer.add_scalar('Round/TrainingSteps', self.step_count, round_num)
            cache_total = self.training_stats['cache_hits'] + self.training_stats['cache_misses']
            if cache_total > 0:
                self.writer.add_scalar(
                    'Round/CacheHitRatio',
                    self.training_stats['cache_hits'] / cache_total,
                    round_num
                )
        
        # Update best hit rate for best model tracking
        if hit_rate > self.best_hit_rate:
            self.best_hit_rate = hit_rate
    
    def save_checkpoint_if_needed(self, round_num: int, hit_rate: float, experiment_metadata: Dict = None):
        """
        Save checkpoint if it's time (based on frequency) and manage checkpoint cleanup.
        Also saves best model if performance improved.
        
        Args:
            round_num: Current round number
            hit_rate: Current hit rate (for best model tracking)
            experiment_metadata: Optional metadata dict (hyperparams, seed, timestamp, etc.)
        """
        if self.checkpoint_dir is None:
            return
            
        import os
        import glob
        from datetime import datetime
        
        # Periodic checkpoint
        if round_num % self.checkpoint_frequency == 0 and round_num > self.last_checkpoint_round:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_round_{round_num}.pth')
            
            # Add timestamp to metadata
            metadata = experiment_metadata.copy() if experiment_metadata else {}
            metadata.update({
                'round': round_num,
                'timestamp': datetime.now().isoformat(),
                'hit_rate': hit_rate,
                'epsilon': self.epsilon,
                'step_count': self.step_count
            })
            
            self.save_model(checkpoint_path, metadata=metadata)
            self.last_checkpoint_round = round_num
            
            # Clean up old checkpoints (keep only last N)
            self._cleanup_old_checkpoints()
        
        # Best model checkpoint (save whenever hit rate improves)
        if hit_rate > self.best_hit_rate:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            metadata = experiment_metadata.copy() if experiment_metadata else {}
            metadata.update({
                'round': round_num,
                'timestamp': datetime.now().isoformat(),
                'hit_rate': hit_rate,
                'best': True,
                'epsilon': self.epsilon,
                'step_count': self.step_count
            })
            self.save_model(best_path, metadata=metadata)
            self.best_model_path = best_path
    
    def _cleanup_old_checkpoints(self):
        """Remove old periodic checkpoints, keeping only the last N"""
        if self.checkpoint_dir is None:
            return
            
        import os
        import glob
        import re
        
        # Find all periodic checkpoint files
        pattern = os.path.join(self.checkpoint_dir, 'checkpoint_round_*.pth')
        checkpoints = glob.glob(pattern)
        
        # Sort by round number (extract from filename)
        def get_round_num(path):
            match = re.search(r'round_(\d+)', path)
            return int(match.group(1)) if match else 0
        
        checkpoints.sort(key=get_round_num, reverse=True)
        
        # Keep only the last N checkpoints
        for old_checkpoint in checkpoints[self.keep_checkpoints:]:
            try:
                os.remove(old_checkpoint)
            except Exception as e:
                print(f"Warning: Could not remove old checkpoint {old_checkpoint}: {e}")
    
    def save_final_checkpoint(self, round_num: int, hit_rate: float, experiment_metadata: Dict = None):
        """
        Save final checkpoint at end of training (always called, regardless of frequency)
        
        Args:
            round_num: Final round number
            hit_rate: Final hit rate
            experiment_metadata: Optional metadata dict
        """
        if self.checkpoint_dir is None:
            return
            
        import os
        from datetime import datetime
        
        final_path = os.path.join(self.checkpoint_dir, 'final_model.pth')
        metadata = experiment_metadata.copy() if experiment_metadata else {}
        metadata.update({
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'hit_rate': hit_rate,
            'final': True,
            'epsilon': self.epsilon,
            'step_count': self.step_count
        })
        self.save_model(final_path, metadata=metadata)
    
    def get_learning_curve(self) -> Dict:
        """Get learning curve data for analysis"""
        return self.learning_curve.copy()
    
    def export_learning_curve(self, filepath: str):
        """Export learning curve to JSON file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.learning_curve, f, indent=2)
        
    def set_checkpoint_config(self, checkpoint_dir: str, frequency: int = 10, keep_last: int = 5):
        """
        Configure checkpoint saving for research experiments
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            frequency: Save checkpoint every N rounds (default: 10)
            keep_last: Keep last N checkpoints, delete older ones (default: 5)
        """
        import os
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_frequency = frequency
        self.keep_checkpoints = keep_last
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def save_model(self, path: str, metadata: Dict = None):
        """
        Save model parameters and training state with optional metadata
        
        Args:
            path: Path to save checkpoint
            metadata: Optional dict with experiment metadata (hyperparams, seed, etc.)
        """
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'training_stats': self.training_stats,
            'learning_curve': self.learning_curve,
            'best_reward': self.best_reward,
            'best_hit_rate': self.best_hit_rate,
        }
        
        # Add metadata if provided (for reproducibility)
        if metadata:
            checkpoint['metadata'] = metadata
            
        torch.save(checkpoint, path)
        
    def load_model(self, path: str):
        """Load model parameters and training state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.training_stats = checkpoint['training_stats']
        
        # Load additional state if available
        if 'learning_curve' in checkpoint:
            self.learning_curve = checkpoint['learning_curve']
        if 'best_reward' in checkpoint:
            self.best_reward = checkpoint['best_reward']
        if 'best_hit_rate' in checkpoint:
            self.best_hit_rate = checkpoint['best_hit_rate']
            
        return checkpoint.get('metadata', {})
        
    def close(self):
        """Clean up resources"""
        if self.writer is not None:
            self.writer.close()
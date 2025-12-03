"""
Gym-style environment wrapper for MARL experiments on the NDN simulator.
Provides reset/step APIs, per-router observations, and joint rewards so that
multi-agent algorithms can plug in without modifying the core simulator.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import gym
    from gym import spaces
except ImportError:  # pragma: no cover - gym optional
    gym = None
    spaces = None

from benchmark import load_scenario_config
from main import create_network, run_simulation
from metrics import get_metrics_collector


DEFAULT_ACTION_MAP: Dict[int, Tuple[str, str]] = {
    0: ("lru", "lce"),
    1: ("lfu", "prob_cache"),
    2: ("fifo", "lcd"),
    3: ("combined", "prob_cache"),
    4: ("lfu", "prob_cache_lfu"),
    5: ("2-lru", "pop_edge"),
    6: ("combined", "mpc_cache"),
}


class MarlNdncachingEnv(gym.Env if gym else object):
    """Multi-agent environment exposing per-router observations and rewards."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        scenario: str = "medium",
        scenario_path: Optional[str] = None,
        rounds_per_step: int = 1,
        rebuild_on_reset: bool = False,
        action_map: Optional[Dict[int, Tuple[str, str]]] = None,
        metrics_dir: Optional[str] = None,
        enable_dqn: bool = False,
    ):
        self.scenario_name = scenario
        self.scenario_path = scenario_path
        self.rounds_per_step = max(1, rounds_per_step)
        self.rebuild_on_reset = rebuild_on_reset
        self.action_map = action_map or DEFAULT_ACTION_MAP
        self.metrics_dir = Path(metrics_dir) if metrics_dir else None

        self.config = load_scenario_config(scenario_path or scenario)
        if not enable_dqn:
            self.config["NDN_SIM_USE_DQN"] = 0
        self.requests_per_round = int(self.config.get("NDN_SIM_REQUESTS", 10))
        self.metrics_collector = get_metrics_collector()

        self.G = None
        self.users = []
        self.producers = []
        self.runtime = None
        self.current_step = 0
        self._prev_router_rates: Dict[int, float] = {}
        self._prev_router_utils: Dict[int, float] = {}

        self._build_network()
        self.observation_dim = 8
        if spaces:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(self.observation_dim,), dtype=np.float32
            )
            self.action_space = spaces.Discrete(len(self.action_map))

    def _build_network(self):
        """Create network topology using scenario config."""
        for key, value in self.config.items():
            os.environ[key] = str(value)
        num_nodes = int(self.config.get("NDN_SIM_NODES", 30))
        num_producers = int(self.config.get("NDN_SIM_PRODUCERS", 6))
        num_contents = int(self.config.get("NDN_SIM_CONTENTS", 200))
        num_users = int(self.config.get("NDN_SIM_USERS", 20))

        self.G, self.users, self.producers, self.runtime = create_network(
            num_nodes=num_nodes,
            num_producers=num_producers,
            num_contents=num_contents,
            num_users=num_users,
            cache_policy=self.config.get("NDN_SIM_CACHE_POLICY", "lru"),
        )

    def reset(self):
        """Reset environment state and optionally rebuild topology."""
        if self.rebuild_on_reset or self.G is None:
            self._build_network()
        self.metrics_collector.reset()
        self.current_step = 0
        self._prev_router_rates.clear()
        self._prev_router_utils.clear()
        return self._collect_observations()

    def step(self, actions: Dict[int, int]):
        """Apply per-router actions, run one simulation block, and return stats."""
        self.current_step += 1
        self.metrics_collector.reset()

        for node, data in self.G.nodes(data=True):
            if "router" not in data:
                continue
            router = data["router"]
            if router.capacity <= 0:
                continue
            action_idx = actions.get(router.router_id, 0)
            replacement, placement = self.action_map.get(action_idx, self.action_map[0])
            if hasattr(router, "content_store"):
                router.content_store.set_replacement_policy(replacement)
            router.set_placement_policy(placement)

        stats = run_simulation(
            self.G,
            self.users,
            self.producers,
            num_rounds=self.rounds_per_step,
            num_requests=self.requests_per_round,
        )

        round_metrics = self.metrics_collector.export_round_metrics(
            self.current_step, self.metrics_dir
        )
        rewards = self._compute_rewards(round_metrics)
        observations = self._collect_observations()
        done = False
        info = {"network_stats": stats, "round_metrics": round_metrics}
        return observations, rewards, done, info

    def _compute_rewards(self, round_metrics: Dict) -> Dict[int, float]:
        rewards: Dict[int, float] = {}
        per_router_rates = round_metrics.get("per_router_hit_rate", {})
        for router_id, rate in per_router_rates.items():
            prev = self._prev_router_rates.get(router_id, 0.0)
            rewards[router_id] = rate - prev
            self._prev_router_rates[router_id] = rate
        per_router_utils = round_metrics.get("per_router_cache_utilization", {})
        for router_id, util in per_router_utils.items():
            prev_util = self._prev_router_utils.get(router_id, 0.0)
            delta = (util - prev_util) / 100.0
            if delta != 0.0:
                rewards[router_id] = rewards.get(router_id, 0.0) + delta * 0.5
            self._prev_router_utils[router_id] = util
        global_hit_rate = round_metrics.get("cache_hit_rate", {}).get("hit_rate", 0.0)
        rewards["global"] = global_hit_rate / 100.0
        return rewards

    def _collect_observations(self) -> Dict[int, np.ndarray]:
        obs: Dict[int, np.ndarray] = {}
        per_router_rates = self.metrics_collector.get_per_router_hit_rates()
        for node, data in self.G.nodes(data=True):
            if "router" not in data:
                continue
            router = data["router"]
            if router.capacity <= 0:
                continue
        obs[router.router_id] = self._router_observation(
            router, per_router_rates.get(router.router_id, 0.0)
        )
        return obs

    def _router_observation(self, router, recent_hit_rate: float) -> np.ndarray:
        cs = router.content_store
        remaining = getattr(cs, "remaining_capacity", 0)
        total = getattr(cs, "total_capacity", 1)
        occupancy = 1.0 - (remaining / max(1, total))
        cache_size = len(getattr(cs, "store", {})) / max(1, total)
        insertion_attempts = getattr(router.stats, "cache_insertion_attempts", 0)
        insertion_successes = getattr(router.stats, "cache_insertion_successes", 0)
        insertion_rate = (
            insertion_successes / insertion_attempts if insertion_attempts else 0.0
        )
        graph_ref = self.G
        neighbors = getattr(router, "neighbors", [])
        neighbor_util = 0.0
        neighbor_samples = 0
        if graph_ref is not None:
            for neighbor_id in neighbors:
                neighbor_data = graph_ref.nodes.get(neighbor_id, {})
                neighbor_router = neighbor_data.get("router")
                if neighbor_router and hasattr(neighbor_router, "content_store"):
                    n_cs = neighbor_router.content_store
                    total_cap = getattr(n_cs, "total_capacity", 0)
                    if total_cap > 0:
                        used = total_cap - getattr(n_cs, "remaining_capacity", 0)
                        neighbor_util += used / total_cap
                        neighbor_samples += 1
        neighbor_util = neighbor_util / max(1, neighbor_samples)
        hop_distance = cs._estimate_hop_distance_to_producer() if hasattr(cs, "_estimate_hop_distance_to_producer") else 0
        hop_closeness = 1.0 / (1.0 + hop_distance)
        pit_entries = getattr(getattr(router, "PIT", None), "entries", {}) if hasattr(router, "PIT") else {}
        pit_load = len(pit_entries) if pit_entries else 0
        pit_feature = min(1.0, pit_load / max(1, len(neighbors) + 1))
        mode_flag = 1.0 if getattr(cs, "mode", "basic") == "dqn_cache" else 0.0
        return np.array(
            [
                occupancy,
                cache_size,
                recent_hit_rate,
                insertion_rate,
                neighbor_util,
                hop_closeness,
                pit_feature,
                mode_flag,
            ],
            dtype=np.float32,
        )

    def render(self, mode="human"):
        print(f"[Env] Step {self.current_step}")

    def close(self):
        if self.runtime is not None:
            self.runtime.shutdown()


def make_marl_env(**kwargs) -> MarlNdncachingEnv:
    """Factory helper used by training scripts."""
    return MarlNdncachingEnv(**kwargs)


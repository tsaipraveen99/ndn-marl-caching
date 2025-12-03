import math
import os
import random
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from router import Router
    from packet import Data


class BasePlacementPolicy:
    """Base class for caching placement decisions."""
    name: str = "lce"

    def should_cache(self, router: "Router", data: "Data", prev_node: int) -> bool:
        return True

    def record_cache_result(self, router: "Router", data: "Data", success: bool) -> None:
        """Hook allowing policies to update packet metadata after caching."""
        return


class LeaveCopyEverywhere(BasePlacementPolicy):
    name = "lce"


class NoCachePolicy(BasePlacementPolicy):
    name = "none"

    def should_cache(self, router: "Router", data: "Data", prev_node: int) -> bool:
        return False


class EdgeOnlyPolicy(BasePlacementPolicy):
    name = "edge_only"

    def should_cache(self, router: "Router", data: "Data", prev_node: int) -> bool:
        # Cache only on user routers or routers directly attached to users
        if router.type == "user":
            return True
        return bool(getattr(router, "connected_users", []))


class LeaveCopyDownPolicy(BasePlacementPolicy):
    name = "lcd"

    def should_cache(self, router: "Router", data: "Data", prev_node: int) -> bool:
        return not getattr(data, "cached_downstream", False)

    def record_cache_result(self, router: "Router", data: "Data", success: bool) -> None:
        if success:
            # Only mark as cached downstream if:
            # 1. Router is not a producer (producers are sources, not downstream)
            # 2. Data has traveled at least one hop (current_hops > 0)
            is_producer = getattr(router, 'type', None) == 'producer'
            has_traveled = getattr(data, 'current_hops', 0) > 0
            
            if not is_producer or has_traveled:
                data.cached_downstream = True
                data.last_cached_router = router.router_id


class LeaveCopyProbPolicy(BasePlacementPolicy):
    name = "lcp"

    def __init__(self, probability: Optional[float] = None):
        self.default_probability = probability or float(os.environ.get("NDN_SIM_LCP_PROB", 0.5))

    def should_cache(self, router: "Router", data: "Data", prev_node: int) -> bool:
        if getattr(data, "cached_downstream", False):
            return False
        return random.random() < self.default_probability

    def record_cache_result(self, router: "Router", data: "Data", success: bool) -> None:
        if success:
            data.cached_downstream = True
            data.last_cached_router = router.router_id


class ProbCachePolicy(BasePlacementPolicy):
    name = "prob_cache"

    def should_cache(self, router: "Router", data: "Data", prev_node: int) -> bool:
        # Estimate path length using hop limit or data trace
        if data.hop_limit <= 0:
            return False
        distance_travelled = max(1, data.current_hops)
        path_estimate = max(distance_travelled, len(getattr(data, "data_trace", [])) + 1)
        # Higher probability near consumers, lower near producers
        probability = max(0.05, 1.0 - (distance_travelled / path_estimate))
        # Weight by suggestion hint if provided
        probability *= max(0.1, min(1.0, getattr(data, "suggestion", 0.5)))
        probability = min(1.0, probability)
        return random.random() < probability


class ProbCacheLFUPolicy(BasePlacementPolicy):
    name = "prob_cache_lfu"

    def __init__(self):
        self._prob_cache = ProbCachePolicy()

    def should_cache(self, router: "Router", data: "Data", prev_node: int) -> bool:
        if not self._prob_cache.should_cache(router, data, prev_node):
            return False
        cs = getattr(router, "content_store", None)
        if cs is None or not hasattr(cs, "access_count"):
            return False
        content_name = str(data.name)
        total = sum(cs.access_count.values()) if cs.access_count else 0
        frequency = cs.access_count.get(content_name, 0) / max(1, total)
        threshold = float(os.environ.get("NDN_SIM_PROB_CACHE_LFU_THRESHOLD", "0.05"))
        return frequency >= threshold


class EdgePopularityPolicy(BasePlacementPolicy):
    name = "pop_edge"

    def should_cache(self, router: "Router", data: "Data", prev_node: int) -> bool:
        connected_users = getattr(router, "connected_users", [])
        if not connected_users:
            return False
        cs = getattr(router, "content_store", None)
        content_name = str(data.name)
        frequency = 0.0
        if cs is not None and hasattr(cs, "access_count"):
            total = sum(cs.access_count.values()) if cs.access_count else 0
            frequency = cs.access_count.get(content_name, 0) / max(1, total)
        threshold = float(os.environ.get("NDN_SIM_EDGE_POP_THRESHOLD", "0.02"))
        return frequency >= threshold or random.random() < 0.3


class MPCHeuristicPolicy(BasePlacementPolicy):
    name = "mpc_cache"

    def should_cache(self, router: "Router", data: "Data", prev_node: int) -> bool:
        if getattr(data, "cached_downstream", False):
            return False
        pit_entries = getattr(getattr(router, "PIT", None), "entries", {})
        demand = len(pit_entries.get(str(data.name), [])) if pit_entries else 0
        neighbor_count = len(getattr(router, "neighbors", []))
        demand_threshold = int(os.environ.get("NDN_SIM_MPC_DEMAND_THRESHOLD", "1"))
        degree_threshold = int(os.environ.get("NDN_SIM_MPC_DEGREE_THRESHOLD", "4"))
        return demand >= demand_threshold or neighbor_count >= degree_threshold


class PlacementPolicyManager:
    """Registry for placement policies."""

    def __init__(self):
        self._policies: Dict[str, BasePlacementPolicy] = {}
        self.register(LeaveCopyEverywhere())
        self.register(NoCachePolicy())
        self.register(EdgeOnlyPolicy())
        self.register(LeaveCopyDownPolicy())
        self.register(LeaveCopyProbPolicy())
        self.register(ProbCachePolicy())
        self.register(ProbCacheLFUPolicy())
        self.register(EdgePopularityPolicy())
        self.register(MPCHeuristicPolicy())

    def register(self, policy: BasePlacementPolicy):
        self._policies[policy.name] = policy

    def get(self, name: str) -> BasePlacementPolicy:
        normalized = (name or "lce").lower()
        return self._policies.get(normalized, self._policies["lce"])

    def should_cache(self, router: "Router", data: "Data", prev_node: int, policy_name: str) -> bool:
        policy = self.get(policy_name)
        return policy.should_cache(router, data, prev_node)

    def record_result(self, router: "Router", data: "Data", policy_name: str, success: bool) -> None:
        policy = self.get(policy_name)
        policy.record_cache_result(router, data, success)


placement_policy_manager = PlacementPolicyManager()


def should_cache_content(router: "Router", data: "Data", prev_node: int, policy_name: str) -> bool:
    return placement_policy_manager.should_cache(router, data, prev_node, policy_name)


def record_cache_result(router: "Router", data: "Data", policy_name: str, success: bool) -> None:
    placement_policy_manager.record_result(router, data, policy_name, success)


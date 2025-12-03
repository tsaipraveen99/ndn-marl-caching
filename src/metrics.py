"""
Task 3.1: Comprehensive Evaluation Metrics
Implements latency, redundancy, dispersion, stretch, and other metrics from research report
"""

import time
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import logging
import json
import csv
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Comprehensive metrics collector for NDN simulation evaluation
    """
    
    def __init__(self):
        self.lock = threading.Lock()
        
        # Latency metrics: time from Interest to Data arrival
        self.interest_times: Dict[str, float] = {}  # interest_id -> timestamp
        self.latencies: List[float] = []  # List of latency measurements
        self.latency_by_content: Dict[str, List[float]] = defaultdict(list)
        
        # Content redundancy: duplicate content across routers
        self.content_locations: Dict[str, Set[int]] = defaultdict(set)  # content_name -> set of router_ids
        self.redundancy_counts: List[int] = []  # Number of copies per content
        
        # Interest packet dispersion: how Interests spread through network
        self.interest_paths: Dict[str, List[int]] = {}  # interest_id -> list of router_ids visited
        self.dispersion_metrics: List[int] = []  # Number of unique routers per Interest
        
        # Stretch: actual hops vs optimal hops
        self.interest_hops: Dict[str, int] = {}  # interest_id -> actual hops
        self.optimal_hops: Dict[str, int] = {}  # content_name -> optimal hops (shortest path)
        self.stretch_ratios: List[float] = []
        
        # Hop-based latency metrics
        self.total_hops: List[int] = []  # Total hops (Interest + Data) per satisfied request
        self.hops_by_content: Dict[str, List[int]] = defaultdict(list)
        
        # Cache hit rate: per-router and network-wide
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.hits_by_router: Dict[int, int] = defaultdict(int)
        self.misses_by_router: Dict[int, int] = defaultdict(int)
        
        # Cache utilization: percentage of cache capacity used
        self.cache_utilization: Dict[int, float] = {}  # router_id -> utilization percentage
        self.cache_usage_bytes: Dict[int, Tuple[int, int]] = {}  # router_id -> (used, total)
        
        # Bandwidth metrics: track bytes transferred
        self.interest_bytes: int = 0  # Total bytes for Interest packets
        self.data_bytes: int = 0  # Total bytes for Data packets
        self.bytes_by_content: Dict[str, int] = defaultdict(int)  # Bytes per content
        self.bytes_by_router: Dict[int, int] = defaultdict(int)  # Bytes per router
        
        # Network topology for optimal path calculation
        self.network_graph = None
        self.round_history: List[Dict] = []
        
    def set_network_graph(self, graph):
        """Set network graph for optimal path calculations"""
        self.network_graph = graph
    
    def record_interest(self, interest_id: str, content_name: str, router_id: int, interest_size: int = 0):
        """
        Record when an Interest packet is issued
        
        Args:
            interest_id: Unique identifier for the Interest
            content_name: Name of requested content
            router_id: Router where Interest originated
            interest_size: Size of Interest packet in bytes (for bandwidth tracking)
        """
        with self.lock:
            current_time = time.time()
            self.interest_times[interest_id] = current_time
            if interest_id not in self.interest_paths:
                self.interest_paths[interest_id] = []
            self.interest_paths[interest_id].append(router_id)
            self.interest_hops[interest_id] = 0  # Initialize hop count
            
            # Track bandwidth (Interest packet size)
            if interest_size > 0:
                self.interest_bytes += interest_size
                self.bytes_by_router[router_id] += interest_size
    
    def record_interest_hop(self, interest_id: str, router_id: int):
        """
        Record a hop for an Interest packet
        
        Args:
            interest_id: Unique identifier for the Interest
            router_id: Router where Interest is processed
        """
        with self.lock:
            if interest_id in self.interest_hops:
                self.interest_hops[interest_id] += 1
            if interest_id in self.interest_paths:
                if router_id not in self.interest_paths[interest_id]:
                    self.interest_paths[interest_id].append(router_id)
    
    def record_data_arrival(self, interest_id: str, content_name: str, router_id: int, 
                           from_cache: bool = False, data_size: int = 0, data_hops: int = 0):
        """
        Record when Data packet arrives
        
        Args:
            interest_id: Unique identifier for the Interest
            content_name: Name of content
            router_id: Router where Data arrived
            from_cache: Whether Data came from cache (cache hit)
            data_size: Size of Data packet in bytes (for bandwidth tracking)
        """
        with self.lock:
            if interest_id in self.interest_times:
                # Calculate latency
                latency = time.time() - self.interest_times[interest_id]
                self.latencies.append(latency)
                self.latency_by_content[content_name].append(latency)
                
                # Record cache hit/miss
                if from_cache:
                    self.cache_hits += 1
                    self.hits_by_router[router_id] += 1
                else:
                    self.cache_misses += 1
                    self.misses_by_router[router_id] += 1
                
                # Track hop-based latency (Interest hops + Data hops)
                interest_hops = self.interest_hops.get(interest_id, 0)
                total_hops = interest_hops + max(0, data_hops)
                self.total_hops.append(total_hops)
                self.hops_by_content[content_name].append(total_hops)
                
                # Track bandwidth (Data packet size)
                if data_size > 0:
                    self.data_bytes += data_size
                    self.bytes_by_content[content_name] += data_size
                    self.bytes_by_router[router_id] += data_size
                
                # Calculate stretch if we have network graph
                if self.network_graph and interest_id in self.interest_hops:
                    actual_hops = self.interest_hops[interest_id]
                    # Find optimal hops (shortest path from origin to producer)
                    # This is a simplified calculation
                    if content_name in self.optimal_hops:
                        optimal = self.optimal_hops[content_name]
                    else:
                        # Estimate optimal as average shortest path in network
                        optimal = self._estimate_optimal_hops()
                        self.optimal_hops[content_name] = optimal
                    
                    if optimal > 0:
                        stretch = actual_hops / optimal
                        self.stretch_ratios.append(stretch)
                
                # Calculate dispersion
                if interest_id in self.interest_paths:
                    unique_routers = len(set(self.interest_paths[interest_id]))
                    self.dispersion_metrics.append(unique_routers)
                
                # Clean up
                del self.interest_times[interest_id]
                if interest_id in self.interest_paths:
                    del self.interest_paths[interest_id]
                if interest_id in self.interest_hops:
                    del self.interest_hops[interest_id]
    
    def record_content_location(self, content_name: str, router_id: int):
        """
        Record where content is cached
        
        Args:
            content_name: Name of content
            router_id: Router where content is cached
        """
        with self.lock:
            self.content_locations[content_name].add(router_id)
    
    def record_cache_utilization(self, router_id: int, used: int, total: int):
        """
        Record cache utilization for a router
        
        Args:
            router_id: Router ID
            used: Used cache capacity
            total: Total cache capacity
        """
        with self.lock:
            if total > 0:
                utilization = (used / total) * 100.0
                self.cache_utilization[router_id] = utilization
                self.cache_usage_bytes[router_id] = (used, total)
    
    def _estimate_optimal_hops(self) -> float:
        """Estimate optimal number of hops (simplified)"""
        if self.network_graph is None:
            return 1.0
        try:
            import networkx as nx
            # Calculate average shortest path length
            if hasattr(nx, 'average_shortest_path_length'):
                try:
                    return nx.average_shortest_path_length(self.network_graph)
                except:
                    return 3.0  # Default estimate
            return 3.0
        except:
            return 3.0
    
    def get_latency_metrics(self) -> Dict:
        """Get latency statistics"""
        with self.lock:
            if not self.latencies:
                return {
                    'mean': 0.0,
                    'median': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'std': 0.0,
                    'p95': 0.0,
                    'p99': 0.0,
                    'count': 0
                }
            
            latencies = np.array(self.latencies)
            return {
                'mean': float(np.mean(latencies)),
                'median': float(np.median(latencies)),
                'min': float(np.min(latencies)),
                'max': float(np.max(latencies)),
                'std': float(np.std(latencies)),
                'p95': float(np.percentile(latencies, 95)),
                'p99': float(np.percentile(latencies, 99)),
                'count': len(self.latencies)
            }
    
    def get_hop_latency_metrics(self) -> Dict:
        """Get hop-based latency statistics"""
        with self.lock:
            if not self.total_hops:
                return {
                    'mean': 0.0,
                    'median': 0.0,
                    'min': 0,
                    'max': 0,
                    'std': 0.0,
                    'p95': 0.0,
                    'p99': 0.0,
                    'count': 0
                }
            
            hops_array = np.array(self.total_hops)
            return {
                'mean': float(np.mean(hops_array)),
                'median': float(np.median(hops_array)),
                'min': int(np.min(hops_array)),
                'max': int(np.max(hops_array)),
                'std': float(np.std(hops_array)),
                'p95': float(np.percentile(hops_array, 95)),
                'p99': float(np.percentile(hops_array, 99)),
                'count': len(self.total_hops)
            }
    
    def get_redundancy_metrics(self) -> Dict:
        """Get content redundancy statistics"""
        with self.lock:
            if not self.content_locations:
                return {
                    'mean': 0.0,
                    'median': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'total_contents': 0
                }
            
            redundancy_counts = [len(locations) for locations in self.content_locations.values()]
            redundancy_array = np.array(redundancy_counts)
            
            return {
                'mean': float(np.mean(redundancy_array)),
                'median': float(np.median(redundancy_array)),
                'min': int(np.min(redundancy_array)),
                'max': int(np.max(redundancy_array)),
                'total_contents': len(self.content_locations)
            }
    
    def get_dispersion_metrics(self) -> Dict:
        """Get Interest packet dispersion statistics"""
        with self.lock:
            if not self.dispersion_metrics:
                return {
                    'mean': 0.0,
                    'median': 0.0,
                    'min': 0,
                    'max': 0,
                    'count': 0
                }
            
            dispersion_array = np.array(self.dispersion_metrics)
            return {
                'mean': float(np.mean(dispersion_array)),
                'median': float(np.median(dispersion_array)),
                'min': int(np.min(dispersion_array)),
                'max': int(np.max(dispersion_array)),
                'count': len(self.dispersion_metrics)
            }
    
    def get_stretch_metrics(self) -> Dict:
        """Get stretch ratio statistics"""
        with self.lock:
            if not self.stretch_ratios:
                return {
                    'mean': 0.0,
                    'median': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'count': 0
                }
            
            stretch_array = np.array(self.stretch_ratios)
            return {
                'mean': float(np.mean(stretch_array)),
                'median': float(np.median(stretch_array)),
                'min': float(np.min(stretch_array)),
                'max': float(np.max(stretch_array)),
                'count': len(self.stretch_ratios)
            }
    
    def get_cache_hit_rate(self) -> Dict:
        """Get cache hit rate statistics"""
        with self.lock:
            total_requests = self.cache_hits + self.cache_misses
            if total_requests == 0:
                return {
                    'hit_rate': 0.0,
                    'miss_rate': 0.0,
                    'total_requests': 0,
                    'hits': 0,
                    'misses': 0
                }
            
            hit_rate = (self.cache_hits / total_requests) * 100.0
            miss_rate = (self.cache_misses / total_requests) * 100.0
            
            return {
                'hit_rate': hit_rate,
                'miss_rate': miss_rate,
                'total_requests': total_requests,
                'hits': self.cache_hits,
                'misses': self.cache_misses
            }
    
    def get_cache_utilization_stats(self) -> Dict:
        """Get cache utilization statistics"""
        with self.lock:
            if not self.cache_utilization:
                return {
                    'mean': 0.0,
                    'median': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'count': 0
                }
            
            utilizations = list(self.cache_utilization.values())
            util_array = np.array(utilizations)
            
            return {
                'mean': float(np.mean(util_array)),
                'median': float(np.median(util_array)),
                'min': float(np.min(util_array)),
                'max': float(np.max(util_array)),
                'count': len(utilizations)
            }

    def get_per_router_hit_rates(self) -> Dict[int, float]:
        """Return per-router hit rates."""
        with self.lock:
            rates: Dict[int, float] = {}
            router_ids = set(self.hits_by_router.keys()) | set(self.misses_by_router.keys())
            for router_id in router_ids:
                hits = self.hits_by_router.get(router_id, 0)
                misses = self.misses_by_router.get(router_id, 0)
                total = hits + misses
                if total > 0:
                    rates[router_id] = hits / total
            return rates

    def export_round_metrics(self, round_number: int, output_dir: Optional[Path] = None) -> Dict:
        """Persist a snapshot of metrics for the current round."""
        summary = {
            'round': round_number,
            'timestamp': time.time(),
            'latency': self.get_latency_metrics(),
            'hop_latency': self.get_hop_latency_metrics(),
            'redundancy': self.get_redundancy_metrics(),
            'dispersion': self.get_dispersion_metrics(),
            'stretch': self.get_stretch_metrics(),
            'cache_hit_rate': self.get_cache_hit_rate(),
            'cache_utilization': self.get_cache_utilization_stats(),
            'bandwidth': self.get_bandwidth_metrics(),
            'fairness': self.get_fairness_metrics(),
            'per_router_hit_rate': self.get_per_router_hit_rates(),
            'per_router_cache_utilization': dict(self.cache_utilization),
            'per_router_cache_bytes': dict(self.cache_usage_bytes)
        }
        with self.lock:
            self.round_history.append(summary)
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            json_path = output_dir / f"metrics_round_{round_number:03d}.json"
            with json_path.open("w") as fp:
                json.dump(summary, fp, indent=2)
            csv_path = output_dir / "metrics_summary.csv"
            self._append_csv_summary(csv_path, summary)
        return summary

    def _append_csv_summary(self, csv_path: Path, summary: Dict):
        """Append a flattened summary row to CSV."""
        header = [
            "round",
            "timestamp",
            "hit_rate",
            "miss_rate",
            "latency_mean",
            "latency_p95",
            "latency_p99",
            "hop_latency_mean",
            "bandwidth_total_bytes",
            "avg_cache_utilization",
            "redundancy_mean"
        ]
        row = {
            "round": summary.get("round"),
            "timestamp": summary.get("timestamp"),
            "hit_rate": summary["cache_hit_rate"].get("hit_rate", 0.0),
            "miss_rate": summary["cache_hit_rate"].get("miss_rate", 0.0),
            "latency_mean": summary["latency"].get("mean", 0.0),
            "latency_p95": summary["latency"].get("p95", 0.0),
            "latency_p99": summary["latency"].get("p99", 0.0),
            "hop_latency_mean": summary["hop_latency"].get("mean", 0.0),
            "bandwidth_total_bytes": summary["bandwidth"].get("total_bytes", 0),
            "avg_cache_utilization": summary["cache_utilization"].get("mean", 0.0),
            "redundancy_mean": summary["redundancy"].get("mean", 0.0),
        }
        file_exists = csv_path.exists()
        with csv_path.open("a", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=header)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    
    def get_bandwidth_metrics(self) -> Dict:
        """Get bandwidth usage statistics"""
        with self.lock:
            total_bytes = self.interest_bytes + self.data_bytes
            
            # Calculate per-router statistics
            router_bytes = list(self.bytes_by_router.values())
            router_bytes_array = np.array(router_bytes) if router_bytes else np.array([0])
            
            return {
                'total_bytes': int(total_bytes),
                'interest_bytes': int(self.interest_bytes),
                'data_bytes': int(self.data_bytes),
                'mean_bytes_per_router': float(np.mean(router_bytes_array)) if len(router_bytes_array) > 0 else 0.0,
                'max_bytes_per_router': float(np.max(router_bytes_array)) if len(router_bytes_array) > 0 else 0.0,
                'min_bytes_per_router': float(np.min(router_bytes_array)) if len(router_bytes_array) > 0 else 0.0,
                'routers_with_traffic': len(self.bytes_by_router),
                'unique_contents': len(self.bytes_by_content)
            }
    
    def get_fairness_metrics(self) -> Dict:
        """
        Get fairness and diversity metrics
        
        Returns:
            Dictionary with:
            - cache_diversity: Number of unique contents cached
            - hit_rate_variance: Variance of per-router hit rates
            - hit_rate_gini: Gini coefficient of cache hits (fairness measure)
            - redundancy: Content distribution across routers
        """
        with self.lock:
            # Cache diversity: number of unique contents cached
            unique_contents = len(self.content_locations)
            
            # Calculate per-router hit rates
            router_hit_rates = []
            for router_id in set(list(self.hits_by_router.keys()) + list(self.misses_by_router.keys())):
                hits = self.hits_by_router.get(router_id, 0)
                misses = self.misses_by_router.get(router_id, 0)
                total = hits + misses
                if total > 0:
                    hit_rate = hits / total
                    router_hit_rates.append(hit_rate)
            
            # Hit rate variance
            hit_rate_variance = float(np.var(router_hit_rates)) if router_hit_rates else 0.0
            
            # Gini coefficient for cache hits (fairness measure)
            # Gini = 1 - 2 * sum of cumulative proportions
            hit_counts = list(self.hits_by_router.values())
            if hit_counts and sum(hit_counts) > 0:
                sorted_hits = sorted(hit_counts)
                n = len(sorted_hits)
                cumsum = np.cumsum(sorted_hits)
                cumsum_normalized = cumsum / cumsum[-1] if cumsum[-1] > 0 else cumsum
                gini = 1.0 - 2.0 * np.mean(cumsum_normalized)
            else:
                gini = 0.0
            
            # Redundancy: mean copies per content
            redundancy_counts_list = [len(loc) for loc in self.content_locations.values()]
            mean_redundancy = float(np.mean(redundancy_counts_list)) if redundancy_counts_list else 0.0
            
            return {
                'cache_diversity': unique_contents,
                'hit_rate_variance': hit_rate_variance,
                'hit_rate_std': float(np.std(router_hit_rates)) if router_hit_rates else 0.0,
                'hit_rate_gini': float(gini),
                'mean_redundancy': mean_redundancy,
                'routers_with_hits': len(self.hits_by_router)
            }
    
    def get_communication_overhead_comparison(self) -> Dict:
        """
        Phase 8.2: Compare communication overhead: Bloom filters vs Fei Wang (exact state)
        
        Returns:
            Dictionary with overhead comparison metrics
        """
        with self.lock:
            # Get Bloom filter overhead (from Interest bytes tagged as bloom_filter_*)
            bloom_filter_bytes = 0
            fei_wang_bytes = 0
            
            # Estimate Bloom filter size (from bandwidth metrics)
            # Bloom filters are tracked as Interest packets with bloom_filter_ prefix
            # For exact comparison, we need to estimate:
            # - Bloom filter size: (size + 7) // 8 bytes per filter
            # - Fei Wang exact state: content_name_size * num_cached_items per neighbor
            
            # Get total Interest bytes (includes Bloom filter propagation)
            total_interest_bytes = self.interest_bytes
            
            # Estimate: Bloom filter propagation is a fraction of Interest bytes
            # In practice, Bloom filters are sent periodically, not per-request
            # Assume Bloom filter size is ~250 bytes (2000 bits / 8 = 250 bytes)
            # And they're sent every N cache operations (e.g., every 10 operations)
            # So Bloom filter overhead ≈ (num_routers * num_neighbors * bloom_filter_size) / propagation_frequency
            
            # For Fei Wang: exact state exchange requires sending full cache contents
            # Each content name is ~50 bytes, and cache has ~10 items (1% of 1000 catalog)
            # So per-neighbor: 50 bytes * 10 items = 500 bytes per update
            # Updates happen more frequently (every cache change)
            
            # Simplified estimation:
            # - Bloom filter: 250 bytes per filter, sent every 10 cache operations
            # - Fei Wang: 500 bytes per neighbor, sent every cache operation
            
            # Get cache statistics to estimate
            total_cached_items = sum(self.cache_utilization.values()) if self.cache_utilization else 0
            num_routers = len(self.cache_utilization) if self.cache_utilization else 0
            
            # Estimate Bloom filter overhead
            # Assume: 250 bytes per filter, sent to all neighbors, every 10 cache operations
            avg_neighbors = 4  # Average neighbors per router (Watts-Strogatz k=4)
            bloom_filter_size = 250  # bytes
            bloom_propagation_freq = 10  # every 10 cache operations
            
            if total_cached_items > 0:
                bloom_filter_updates = total_cached_items / bloom_propagation_freq
                bloom_filter_bytes = bloom_filter_updates * num_routers * avg_neighbors * bloom_filter_size
            
            # Estimate Fei Wang exact state overhead
            # Assume: 50 bytes per content name, 10 items per cache, sent every cache operation
            avg_cache_size = total_cached_items / max(1, num_routers) if num_routers > 0 else 10
            content_name_size = 50  # bytes per content name
            fei_wang_update_size = avg_cache_size * content_name_size  # bytes per update
            
            if total_cached_items > 0:
                fei_wang_updates = total_cached_items  # every cache operation
                fei_wang_bytes = fei_wang_updates * num_routers * avg_neighbors * fei_wang_update_size
            
            # Calculate overhead ratio
            overhead_ratio = bloom_filter_bytes / max(1, fei_wang_bytes) if fei_wang_bytes > 0 else 0.0
            overhead_reduction = (1.0 - overhead_ratio) * 100.0 if overhead_ratio < 1.0 else 0.0
            
            return {
                'bloom_filter_bytes': int(bloom_filter_bytes),
                'fei_wang_bytes': int(fei_wang_bytes),
                'overhead_ratio': float(overhead_ratio),
                'overhead_reduction_percent': float(overhead_reduction),
                'bloom_filter_size_bytes': bloom_filter_size,
                'fei_wang_update_size_bytes': int(fei_wang_update_size),
                'num_routers': num_routers,
                'avg_neighbors': avg_neighbors,
                'total_cached_items': int(total_cached_items)
            }
    
    def get_all_metrics(self) -> Dict:
        """Get all collected metrics"""
        return {
            'latency': self.get_latency_metrics(),
            'hop_latency': self.get_hop_latency_metrics(),
            'redundancy': self.get_redundancy_metrics(),
            'dispersion': self.get_dispersion_metrics(),
            'stretch': self.get_stretch_metrics(),
            'cache_hit_rate': self.get_cache_hit_rate(),
            'cache_utilization': self.get_cache_utilization_stats(),
            'bandwidth': self.get_bandwidth_metrics(),
            'fairness': self.get_fairness_metrics(),
            'communication_overhead': self.get_communication_overhead_comparison()
        }
    
    def export_to_dict(self) -> Dict:
        """Export all metrics to dictionary for JSON serialization"""
        return self.get_all_metrics()
    
    def reset(self):
        """Reset all metrics"""
        with self.lock:
            self.interest_times.clear()
            self.latencies.clear()
            self.latency_by_content.clear()
            self.content_locations.clear()
            self.redundancy_counts.clear()
            self.interest_paths.clear()
            self.dispersion_metrics.clear()
            self.interest_hops.clear()
            self.optimal_hops.clear()
            self.stretch_ratios.clear()
            self.cache_hits = 0
            self.cache_misses = 0
            self.hits_by_router.clear()
            self.misses_by_router.clear()
            self.cache_utilization.clear()
            self.cache_usage_bytes.clear()
            self.interest_bytes = 0
            self.data_bytes = 0
            self.bytes_by_content.clear()
            self.bytes_by_router.clear()
            self.total_hops.clear()
            self.hops_by_content.clear()
            self.round_history.clear()


# Global metrics collector instance
_global_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector"""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
    return _global_metrics_collector


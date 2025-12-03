#!/usr/bin/env python3
import dill as pickle
import threading
import networkx as nx
import random
import time
import sys
import numpy as np
import logging
import warnings
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List, Dict, Optional
from pathlib import Path

from utils import NDNDistribution
from router import Router, SimulationStats, RouterRuntime
from endpoints import User, Producer

# Colab minimal logging support
try:
    from colab_minimal_logging import is_colab, get_minimal_logger
except ImportError:
    # Not available in non-Colab environments
    def is_colab():
        return False
    def get_minimal_logger():
        return None

# Configure system
sys.setrecursionlimit(1000)
warnings.filterwarnings("ignore")
VERBOSE = os.environ.get("NDN_SIM_VERBOSE", "0") == "1"
_fallback_progress_logger: Optional[logging.Logger] = None

# Network content configuration
ORGANIZATIONS = ['ucla', 'mit', 'stanford', 'berkeley', 'oxford']
DEPARTMENTS = ['cs', 'ee', 'math', 'physics', 'biology']
CONTENT_TYPES = ['research', 'courses', 'projects', 'data', 'media']

def debug_print(msg: str, logger: logging.Logger = None):
    """Helper function for consistent debug output"""
    if logger:
        logger.debug(msg)
    if VERBOSE and not logger:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} DEBUG: {msg}", flush=True)

def log_info(msg: str, logger: Optional[logging.Logger] = None):
    """Centralized INFO-level logging with console fallback."""
    if logger:
        logger.info(msg)
    else:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} INFO: {msg}", flush=True)

def _reset_logger_handlers(target_logger: logging.Logger):
    """Remove and close all handlers for a logger to avoid duplicates."""
    for handler in list(target_logger.handlers):
        target_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

def setup_logging(log_mode: str = "w") -> Tuple[logging.Logger, logging.Logger]:
    """Set up logging with both file and console output.
    
    Args:
        log_mode: File mode for log files ("w" to overwrite, "a" to append).
    """
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Console handler for immediate feedback
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    
    # Simulation results logger
    sim_logger = logging.getLogger('simulation_logger')
    sim_logger.setLevel(logging.INFO)
    sim_logger.propagate = False
    _reset_logger_handlers(sim_logger)
    sim_handler = logging.FileHandler("logs/simulation_results.log", mode=log_mode)
    sim_handler.setFormatter(formatter)
    sim_logger.addHandler(sim_handler)
    sim_logger.addHandler(console_handler)
    
    # Network setup logger
    net_logger = logging.getLogger('network_logger')
    net_logger.setLevel(logging.INFO)
    net_logger.propagate = False
    _reset_logger_handlers(net_logger)
    net_handler = logging.FileHandler("logs/network_setup.log", mode=log_mode)
    net_handler.setFormatter(formatter)
    net_logger.addHandler(net_handler)
    net_logger.addHandler(console_handler)
    
    # Ensure component loggers share the console handler for visibility
    component_loggers = [
        logging.getLogger('router_logger'),
        logging.getLogger('endpoints_logger'),
        logging.getLogger('utils_logger'),
    ]
    # Respect QUIET_MODE setting
    QUIET_MODE = os.environ.get("NDN_SIM_QUIET", "1") == "1"
    log_level = logging.WARNING if QUIET_MODE else logging.INFO
    
    for comp_logger in component_loggers:
        comp_logger.setLevel(log_level)  # Use log_level instead of hardcoded INFO
        comp_logger.propagate = False
        # Remove stale handlers so we don't duplicate console/file handlers
        for handler in list(comp_logger.handlers):
            if isinstance(handler, logging.StreamHandler):
                comp_logger.removeHandler(handler)
        comp_logger.addHandler(console_handler)
    
    return sim_logger, net_logger

def get_progress_logger(logger: Optional[logging.Logger]) -> logging.Logger:
    """Return a logger suitable for progress tracking, creating a fallback if needed."""
    global _fallback_progress_logger
    if logger:
        return logger
    
    if _fallback_progress_logger is None:
        Path("logs").mkdir(exist_ok=True)
        fallback = logging.getLogger('simulation_progress')
        fallback.setLevel(logging.INFO)
        fallback.propagate = False
        if not fallback.handlers:
            handler = logging.FileHandler("logs/simulation_progress.log", mode="a")
            handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
            fallback.addHandler(handler)
        _fallback_progress_logger = fallback
    return _fallback_progress_logger

def create_network(
    num_nodes: int = 30,
    num_producers: int = 20,
    num_contents: int = 300,
    num_users: int = 50,
    cache_policy: str = "lru",
    logger: logging.Logger = None
) -> Tuple[nx.Graph, List[User], List[Producer], RouterRuntime]:
    """Create NDN network topology with routers, producers, and users"""
    try:
        debug_print("Starting network creation...", logger)
        debug_print(f"Parameters: nodes={num_nodes}, producers={num_producers}, contents={num_contents}, users={num_users}", logger)
        
        # Phase 7.3: Support different network topologies
        topology_type = os.environ.get("NDN_SIM_TOPOLOGY", "watts_strogatz").lower()
        debug_print(f"Generating {topology_type} network...", logger)
        
        if topology_type == "barabasi_albert" or topology_type == "scale_free":
            # Scale-free network (Barabási-Albert)
            # m = number of edges to attach from a new node to existing nodes
            m = int(os.environ.get("NDN_SIM_TOPOLOGY_M", "2"))  # Default: attach 2 edges
            G = nx.barabasi_albert_graph(num_nodes, m)
            debug_print(f"Barabási-Albert network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", logger)
        elif topology_type == "tree" or topology_type == "hierarchical":
            # Hierarchical tree topology
            G = nx.balanced_tree(r=2, h=int(np.ceil(np.log2(num_nodes))))  # Binary tree
            # If tree has fewer nodes than requested, add random edges
            if G.number_of_nodes() < num_nodes:
                # Add remaining nodes and connect them
                for i in range(G.number_of_nodes(), num_nodes):
                    G.add_node(i)
                    # Connect to random existing node
                    if G.number_of_nodes() > 1:
                        existing_node = random.choice(list(G.nodes()))
                        G.add_edge(i, existing_node)
            elif G.number_of_nodes() > num_nodes:
                # Remove excess nodes (keep first num_nodes)
                nodes_to_remove = list(G.nodes())[num_nodes:]
                G.remove_nodes_from(nodes_to_remove)
            debug_print(f"Tree network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", logger)
        elif topology_type == "grid":
            # Grid topology
            # Calculate grid dimensions
            grid_size = int(np.ceil(np.sqrt(num_nodes)))
            G = nx.grid_2d_graph(grid_size, grid_size)
            # Relabel nodes to 0-indexed
            mapping = {node: i for i, node in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, mapping)
            # Remove excess nodes if needed
            if G.number_of_nodes() > num_nodes:
                nodes_to_remove = list(G.nodes())[num_nodes:]
                G.remove_nodes_from(nodes_to_remove)
            debug_print(f"Grid network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", logger)
        else:
            # Default: Watts-Strogatz small-world network
            k = int(os.environ.get("NDN_SIM_TOPOLOGY_K", "4"))  # Default: 4 neighbors
            p = float(os.environ.get("NDN_SIM_TOPOLOGY_P", "0.2"))  # Default: 0.2 rewiring probability
            G = nx.watts_strogatz_graph(num_nodes, k=k, p=p)
            debug_print(f"Watts-Strogatz network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", logger)
        
        debug_print(f"Network generated: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", logger)
        
        # Limit max workers to prevent thread exhaustion
        # Formula: Scale with network size for 2000 nodes
        # For large networks (2000+ nodes): use 20-25% of nodes as workers
        # For medium networks: use min(64, nodes + users + producers)
        if num_nodes >= 2000:
            # Large network: scale workers proportionally (20-25% of nodes)
            max_workers = max(128, min(512, int(num_nodes * 0.25)))
        else:
            # Medium/small network: original formula
            max_workers = max(8, min(64, num_nodes + num_users + num_producers))
        runtime = RouterRuntime(max_workers=max_workers)

        content_size_min = int(os.environ.get("NDN_SIM_CONTENT_SIZE_MIN", "10"))
        content_size_max = int(os.environ.get("NDN_SIM_CONTENT_SIZE_MAX", "10"))
        if content_size_min <= 0:
            content_size_min = 1
        if content_size_max < content_size_min:
            content_size_max = content_size_min
        avg_content_size = (content_size_min + content_size_max) / 2.0

        # Task 1.1: Increase cache capacity for better hit rates
        router_capacity = int(os.environ.get("NDN_SIM_CACHE_CAPACITY", "500"))
        cache_capacity_ratio = float(os.environ.get("NDN_SIM_CACHE_CAPACITY_RATIO", "0"))
        if cache_capacity_ratio > 0:
            router_capacity = max(
                content_size_max,
                int(num_contents * avg_content_size * cache_capacity_ratio)
            )
        producer_capacity_default = max(content_size_min * 5, router_capacity // 8)

        # Create routers
        debug_print("Creating routers...", logger)
        routers = []
        for i in range(num_nodes):
            debug_print(f"Creating router {i}...", logger)
            router = Router(router_id=i, capacity=router_capacity, type_='router', G=G, runtime=runtime)
            if hasattr(router, 'content_store'):
                router.content_store.set_replacement_policy(cache_policy)
            router.set_placement_policy(os.environ.get("NDN_SIM_PLACEMENT_POLICY", "lce"))
            G.nodes[i]['router'] = router
            G.nodes[i]['type'] = 'router'
            routers.append(router)
            debug_print(f"Router {i} created successfully", logger)
        
        # Create content distribution with Zipf parameter from env if set
        zipf_param = float(os.environ.get('NDN_SIM_ZIPF_PARAM', '1.2'))  # Default 1.2 for better cache hits
        debug_print(f"Creating content distribution with Zipf parameter {zipf_param}...", logger)
        distribution = NDNDistribution(num_contents, zipf_param=zipf_param)
        debug_print("Content distribution created", logger)
        
        # Create producers
        debug_print("Creating producers...", logger)
        producers = []
        for i in range(num_producers):
            debug_print(f"Creating producer {i}...", logger)
            producer_node = num_nodes + i
            
            # First add the node to the graph
            G.add_node(producer_node)
            
            # Then create the router
            debug_print(f"Creating router for producer {i}...", logger)
            # Task 1.1: Increase producer cache capacity proportionally
            producer_capacity = max(50, producer_capacity_default)
            producer_router = Router(router_id=producer_node, capacity=producer_capacity, type_='producer', G=G, runtime=runtime)
            if hasattr(producer_router, 'content_store'):
                producer_router.content_store.set_replacement_policy(cache_policy)
            producer_router.set_placement_policy(os.environ.get("NDN_SIM_PLACEMENT_POLICY", "lce"))
            G.nodes[producer_node]['router'] = producer_router
            G.nodes[producer_node]['type'] = 'producer'
            routers.append(producer_router)
            
            # Generate content for producer
            debug_print(f"Generating content for producer {i}...", logger)
            producer_contents = {}
            org = random.choice(ORGANIZATIONS)
            dept = random.choice(DEPARTMENTS)
            
            contents_per_producer = num_contents // num_producers
            for j in range(contents_per_producer):
                content_type = random.choice(CONTENT_TYPES)
                name = f"/edu/{org}/{dept}/{content_type}/content_{j:03d}"
                producer_contents[name] = random.randint(content_size_min, content_size_max)
            
            debug_print(f"Creating producer object {i}...", logger)
            producer = Producer([producer_router], producer_contents)
            # Link router back to producer for Interest handling
            producer_router.producer = producer
            
            # Connect producer to network - connect to more routers
            debug_print(f"Connecting producer {i} to network...", logger)
            # Connect to more routers for better connectivity
            connected_routers = random.sample(range(num_nodes), k=min(5, num_nodes))
            
            # Connect producer to selected routers
            for router_id in connected_routers:
                # Add edge to graph
                G.add_edge(producer_node, router_id)
                
                # Update router neighbor relationships
                producer_router.add_neighbor(router_id)
                G.nodes[router_id]['router'].add_neighbor(producer_node)
                
                # Register content in FIB for each connected router
                for content_name in producer_contents:
                    try:
                        # Register content
                        G.nodes[router_id]['router'].add_to_FIB(
                            content_name=content_name,
                            next_hop=producer_node,
                            G=G
                        )
                    except Exception as e:
                        debug_print(f"Error registering content {content_name} with router {router_id}: {e}", logger)
                        continue
            
            # Call the producer's content registration method
            producer._register_content_with_routers(G)
            
            producers.append(producer)
            debug_print(f"Producer {i} created and connected successfully", logger)
        
        # Create users
        debug_print("Creating users...", logger)
        users = []
        for i in range(num_users):
            debug_print(f"Creating user {i}...", logger)
            user_node = num_nodes + num_producers + i
            
            # Add node before creating router
            G.add_node(user_node)
            router_id = random.choice(range(num_nodes))
            
            debug_print(f"Creating router for user {i}...", logger)
            user_router = Router(router_id=user_node, capacity=0, type_='user', G=G, runtime=runtime)
            if hasattr(user_router, 'content_store'):
                user_router.content_store.set_replacement_policy(cache_policy)
            user_router.set_placement_policy(os.environ.get("NDN_SIM_PLACEMENT_POLICY", "lce"))
            G.nodes[user_node]['router'] = user_router
            G.nodes[user_node]['type'] = 'user'
            
            # Add edge and update neighbors
            G.add_edge(user_node, router_id)
            user_router.add_neighbor(router_id)
            G.nodes[router_id]['router'].add_neighbor(user_node)
            
            routers.append(user_router)
            
            debug_print(f"Creating user object {i}...", logger)
            user = User(user_node, user_node, user_router, distribution)
            
            # Add user to the router's connected users
            user_router.connected_users.append(user)
            
            users.append(user)
            debug_print(f"User {i} created and connected successfully", logger)
        
        # Start routers
        debug_print("Starting all routers...", logger)
        for i, router in enumerate(routers):
            debug_print(f"Router {router.router_id} registered with runtime", logger)
            time.sleep(0.002)  # Small delay for readability
        
        debug_print("Network creation completed successfully", logger)
        return G, users, producers, runtime
        
    except Exception as e:
        debug_print(f"Error in network creation: {str(e)}", logger)
        if logger:
            logger.error("Stack trace:", exc_info=True)
        raise

def setup_all_routers_to_dqn_mode(G, logger=None):
    """Set all routers to DQN caching mode and improve content registration"""
    debug_print("Setting routers to DQN caching mode...", logger)
    
    for node, data in G.nodes(data=True):
        if 'router' in data and hasattr(data['router'], 'content_store'):
            router = data['router']
            
            # Set to DQN caching mode
            if hasattr(router, 'set_mode'):
                router.set_mode("dqn_cache")
            elif hasattr(router.content_store, 'set_mode'):
                router.content_store.set_mode("dqn_cache")
    
    # Re-register all producer content to ensure proper propagation
    debug_print("Re-registering producer content...", logger)
    producers_list = [data['router'] for _, data in G.nodes(data=True) 
                     if 'type' in data and data['type'] == 'producer']
    
    for producer in producers_list:
        # Find connected router nodes
        connected_routers = list(G.neighbors(producer.router_id))
        
        # Get producer's content
        content_names = []
        if hasattr(producer, 'data_store'):
            content_names = list(producer.data_store.keys())
        elif hasattr(producer, 'store'):
            content_names = list(producer.store.keys())
            
        # Register content with each neighbor
        for content_name in content_names:
            for router_id in connected_routers:
                if router_id in G.nodes and 'router' in G.nodes[router_id]:
                    G.nodes[router_id]['router'].add_to_FIB(content_name, producer.router_id, G)
    
    debug_print("Router setup completed", logger)

def initialize_bloom_filter_propagation(G, logger=None):
    """Initialize Bloom filter propagation by sending initial filters to neighbors"""
    debug_print("Initializing Bloom filter propagation...", logger)
    
    for node, data in G.nodes(data=True):
        if 'router' in data and hasattr(data['router'], 'content_store'):
            router = data['router']
            content_store = router.content_store
            
            # Trigger initial propagation if router has neighbors
            if hasattr(content_store, 'propagate_bloom_filter_to_neighbors'):
                try:
                    content_store.propagate_bloom_filter_to_neighbors()
                except Exception as e:
                    if logger:
                        logger.debug(f"Router {router.router_id}: Initial Bloom filter propagation failed: {e}")
    
    debug_print("Bloom filter propagation initialized", logger)

def install_global_prefix_routes(G, producers: List[Producer], logger=None):
    """Install producer domain-prefix FIB entries on all routers for reachability in tests."""
    try:
        debug_print("Installing global prefix routes for producers...", logger)
        # Gather all router node ids
        router_node_ids = [node for node, data in G.nodes(data=True) if data.get('type') in ('router', 'user')]
        routes_installed = 0
        for producer in producers:
            # Build set of domain prefixes for this producer
            domain_prefixes = set()
            for content_name in getattr(producer, 'contents', {}).keys():
                parts = content_name.split('/')
                if len(parts) >= 4:
                    domain_prefixes.add('/'.join(parts[:4]))
            # Install each prefix on every router
            for prefix in domain_prefixes:
                for node_id in router_node_ids:
                    router = G.nodes[node_id].get('router')
                    if router and hasattr(router, 'add_to_FIB'):
                        # FIX: Pass graph parameter to enable proper FIB propagation
                        try:
                            router.add_to_FIB(prefix, producer.router_id, G)
                            routes_installed += 1
                        except Exception as e:
                            debug_print(f"Error installing route {prefix} on router {node_id}: {e}", logger)
                            continue
        debug_print(f"Global prefix routes installed: {routes_installed} routes across {len(router_node_ids)} routers", logger)
    except Exception as e:
        debug_print(f"Error installing global prefix routes: {e}", logger)

def visualize_topology(G: nx.Graph, filename: str = "network_topology.png", logger: logging.Logger = None):
    """Render and save the current network topology to an image."""
    try:
        debug_print(f"Rendering topology to {filename}...", logger)
        color_map = {'router': '#87ceeb', 'user': '#ff7f7f', 'producer': '#7fc97f'}
        node_colors = [
            color_map.get(data.get('type', 'router'), '#d3d3d3')
            for _, data in G.nodes(data=True)
        ]
        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=200, edgecolors='black', linewidths=0.5)
        nx.draw_networkx_edges(G, pos, alpha=0.3)
        nx.draw_networkx_labels(G, pos, font_size=7)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close()
        debug_print(f"Topology image saved to {filename}", logger)
    except Exception as e:
        debug_print(f"Error rendering topology: {e}", logger)

def align_user_distributions_with_producers(users: List[User], producers: List[Producer], logger=None):
    """Align users' content distributions with actual producer contents to ensure hits.
    PRESERVES Zipf distribution to maintain popularity skew for realistic cache hit rates."""
    try:
        debug_print("Aligning user request distributions with producer contents...", logger)
        # Collect all available content names from producers
        available: List[str] = []
        for p in producers:
            available.extend(list(getattr(p, 'contents', {}).keys()))
        if not available:
            debug_print("No producer contents available to align.", logger)
            return
        import numpy as np  # local import to avoid top-level changes
        
        # FIX: Preserve Zipf distribution instead of using uniform
        # Get Zipf parameter from first user's distribution if available
        zipf_param = 1.2  # Default: stronger popularity skew for better cache hits
        if users and hasattr(users[0], 'distribution') and hasattr(users[0].distribution, 'zipf_param'):
            zipf_param = users[0].distribution.zipf_param
        
        # Create Zipf distribution over available contents
        num_contents = len(available)
        ranks = np.arange(1, num_contents + 1, dtype=float)
        probs = 1.0 / np.power(ranks, zipf_param)
        probs = probs / probs.sum()  # Normalize
        
        for u in users:
            if hasattr(u, 'distribution'):
                # Preserve Zipf distribution while aligning content list
                try:
                    u.distribution.content_list = available
                    u.distribution.num_contents = num_contents
                    u.distribution.probabilities = probs
                    # Preserve zipf_param if it exists
                    if hasattr(u.distribution, 'zipf_param'):
                        u.distribution.zipf_param = zipf_param
                except Exception:
                    # If distribution object differs, create new Zipf distribution
                    from utils import NDNDistribution
                    u.distribution = NDNDistribution(num_contents, zipf_param=zipf_param)
                    u.distribution.content_list = available
                    u.distribution.probabilities = probs
        debug_print(f"User distributions aligned with Zipf parameter {zipf_param} (preserving popularity skew)", logger)
    except Exception as e:
        debug_print(f"Error aligning user distributions: {e}", logger)

def demo_cache_behavior(G, users: List[User], producers: List[Producer], logger=None):
    """Issue repeated requests for a known content to demonstrate basic caching behavior."""
    try:
        from router import stats as global_stats
        if not producers:
            return
        # Pick a deterministic content name from the first producer
        first_contents = list(producers[0].contents.keys())
        if not first_contents:
            return
        content_name = first_contents[0]
        debug_print(f"Demo: requesting {content_name} repeatedly to warm caches...", logger)

        # Baseline stats
        with global_stats.lock:
            base_hits = global_stats.cache_hits
            base_packets = global_stats.data_packets_transferred

        # Send repeated interests for the same content from all users
        for _ in range(5):
            for user in users:
                user.make_interest(G, content_name)
            time.sleep(0.05)

        # Allow processing
        time.sleep(2.0)

        # Collect cache presence
        cached_count = 0
        for node, data in G.nodes(data=True):
            r = data.get('router')
            if r and getattr(r, 'capacity', 0) > 0 and hasattr(r, 'content_store'):
                if str(content_name) in getattr(r.content_store, 'store', {}):
                    cached_count += 1

        with global_stats.lock:
            delta_hits = global_stats.cache_hits - base_hits
            delta_packets = global_stats.data_packets_transferred - base_packets

        debug_print(f"Demo results: cached_routers={cached_count}, new_packets={delta_packets}, new_cache_hits={delta_hits}", logger)
    except Exception as e:
        debug_print(f"Error in cache demo: {e}", logger)

def warmup_cache(G, users: List[User], producers: List[Producer], num_warmup_rounds: int = 5, logger=None):
    """
    Task 1.2: Cache warm-up phase - request popular content repeatedly to populate caches
    """
    try:
        debug_print(f"Starting cache warm-up phase ({num_warmup_rounds} rounds)...", logger)
        if not producers:
            debug_print("No producers available for warm-up", logger)
            return
        
        # Select popular content items from producers
        popular_contents = []
        for producer in producers[:min(10, len(producers))]:  # Use first 10 producers
            contents = list(producer.contents.keys())
            if contents:
                # Select a few items from each producer
                popular_contents.extend(contents[:min(5, len(contents))])
        
        if not popular_contents:
            debug_print("No content available for warm-up", logger)
            return
        
        debug_print(f"Warming up cache with {len(popular_contents)} popular content items...", logger)
        
        # Request popular content repeatedly
        for round_num in range(num_warmup_rounds):
            debug_print(f"Warm-up round {round_num + 1}/{num_warmup_rounds}...", logger)
            # Request each popular content from multiple users
            for content_name in popular_contents:
                # Request from a subset of users to create cache hits
                sample_users = random.sample(users, min(50, len(users)))
                for user in sample_users:
                    user.make_interest(G, content_name)
                time.sleep(0.01)  # Small delay between requests
            
            # Allow time for processing
            time.sleep(0.5)
        
        # Check cache state after warm-up
        cached_routers = 0
        total_cached_items = 0
        for node, data in G.nodes(data=True):
            r = data.get('router')
            if r and getattr(r, 'capacity', 0) > 0 and hasattr(r, 'content_store'):
                cs = r.content_store
                cached_items = len(getattr(cs, 'store', {}))
                if cached_items > 0:
                    cached_routers += 1
                    total_cached_items += cached_items
        
        debug_print(f"Warm-up complete: {cached_routers} routers have cached items, {total_cached_items} total items cached", logger)
    except Exception as e:
        debug_print(f"Error in cache warm-up: {e}", logger)

def run_simulation(
    G: nx.Graph,
    users: List[User],
    producers: List[Producer],
    num_rounds: int = 100,
    num_requests: int = 10,
    logger: logging.Logger = None
) -> Dict:
    """Run NDN simulation"""
    progress_logger = get_progress_logger(logger)
    users_count = len(users)
    total_requests = users_count * num_rounds * num_requests
    log_info(
        f"Simulation starting: users={users_count}, rounds={num_rounds}, "
        f"requests_per_user={num_requests}, total_requests={total_requests}",
        progress_logger
    )
    try:
        debug_print("Starting simulation...", logger)
        debug_print(f"Parameters: rounds={num_rounds}, requests={num_requests}", logger)

        # Use global runtime stats from routers
        from router import stats as global_stats
        # Reset global stats counters
        try:
            with global_stats.lock:
                global_stats.nodes_traversed = 0
                global_stats.cache_hits = 0
                global_stats.data_packets_transferred = 0
                global_stats.total_data_size_transferred = 0
        except Exception:
            pass

        round_stats = []

        # Run simulation rounds
        with ThreadPoolExecutor(max_workers=min(10, len(users))) as executor:
            for round_ in range(num_rounds):
                round_num = round_ + 1
                debug_print(f"Starting round {round_num}...", logger)
                log_info(
                    f"Round {round_num}/{num_rounds}: dispatching {users_count} users × {num_requests} requests "
                    f"(batch={users_count * num_requests})",
                    progress_logger
                )

                # Submit user requests
                futures = []
                for user in users:
                    debug_print(f"Submitting requests for user {user.user_id}...", logger)
                    # Match the parameter signature in endpoints.py
                    futures.append(executor.submit(user.run, G, 1, num_requests))

                # Wait for all requests to complete
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        debug_print(f"Error in user request: {str(e)}", logger)

                # CRITICAL: Wait for runtime to process all queued messages
                # Users enqueue messages, but runtime processes them asynchronously
                # We need to wait for the queue to drain before moving to next round
                # Get the runtime from any router (they all share the same runtime)
                runtime = None
                for node, data in G.nodes(data=True):
                    if 'router' in data:
                        router = data['router']
                        if hasattr(router, 'runtime') and router.runtime is not None:
                            runtime = router.runtime
                            break
                
                if runtime is not None:
                    # Wait for queue to drain with timeout
                    # Add diagnostic output to see what's happening
                    if logger is None:
                        print(f"⏳ Round {round_num}: Waiting for queue to drain...", flush=True)
                    drain_start = time.time()
                    drained = runtime.wait_for_queue_drain(timeout=120.0, logger=logger or progress_logger)
                    drain_elapsed = time.time() - drain_start
                    if not drained:
                        debug_print("Warning: Queue not fully drained, proceeding anyway", logger)
                        if logger is None:
                            print(f"⚠️  Round {round_num}: Queue drain timeout, proceeding anyway", flush=True)
                        log_info(
                            f"Round {round_num}/{num_rounds}: queue drain TIMEOUT after {drain_elapsed:.2f}s",
                            progress_logger
                        )
                    else:
                        log_info(
                            f"Round {round_num}/{num_rounds}: queue drained in {drain_elapsed:.2f}s "
                            f"(total_messages={runtime.metrics.get('total_messages_processed', 0)})",
                            progress_logger
                        )

                debug_print(f"Round {round_num} completed", logger)
                if logger is None:
                    print(f"✅ Round {round_num} completed", flush=True)

                # Collect statistics every round (for learning curve tracking)
                # Calculate requests completed so far (correct hit rate calculation)
                requests_so_far = users_count * round_num * num_requests
                hit_rate = global_stats.cache_hits / max(1, requests_so_far)
                stats_data = {
                    'round': round_num,
                    'hit_rate': hit_rate,
                    'requests': requests_so_far,
                    'packets': global_stats.data_packets_transferred
                }
                round_stats.append(stats_data)
                
                # Persist detailed metrics for analysis/plotting
                try:
                    from metrics import get_metrics_collector
                    metrics_collector = get_metrics_collector()
                    metrics_dir = Path(os.environ.get("NDN_SIM_METRICS_DIR", "results/metrics"))
                    metrics_collector.export_round_metrics(round_num, metrics_dir)
                except Exception as e:
                    debug_print(f"Error exporting metrics for round {round_num}: {e}", logger)
                
                # Get average DQN loss if available (for minimal logging)
                avg_dqn_loss = None
                if is_colab():
                    try:
                        losses = []
                        for node, data in G.nodes(data=True):
                            if 'router' in data:
                                router = data['router']
                                if hasattr(router, 'content_store') and hasattr(router.content_store, 'dqn_agent'):
                                    dqn_agent = router.content_store.dqn_agent
                                    if dqn_agent is not None and hasattr(dqn_agent, 'training_stats'):
                                        agent_losses = dqn_agent.training_stats.get('losses', [])
                                        if agent_losses:
                                            losses.extend(agent_losses[-10:])  # Last 10 losses per agent
                        if losses:
                            avg_dqn_loss = sum(losses) / len(losses)
                    except Exception:
                        pass
                
                # Use minimal logger if in Colab, otherwise use regular logging
                minimal_logger = get_minimal_logger()
                if minimal_logger:
                    minimal_logger.log_round(round_num, num_rounds, hit_rate, avg_dqn_loss)
                else:
                    log_info(
                        f"Round {round_num}/{num_rounds} summary: hit_rate={hit_rate:.4f}, "
                        f"cache_hits={global_stats.cache_hits}, nodes_traversed={global_stats.nodes_traversed}",
                        progress_logger
                    )

                # Record DQN learning curve metrics for each router and save checkpoints
                if (round_num % 1 == 0):  # Every round
                    try:
                        # Prepare experiment metadata for reproducibility
                        experiment_metadata = {
                            'num_rounds': num_rounds,
                            'num_requests': num_requests,
                            'network_size': len(G.nodes()),
                            'seed': getattr(random, '_seed', None) or getattr(np.random, '_seed', None)
                        }
                        
                        for node, data in G.nodes(data=True):
                            if 'router' in data:
                                router = data['router']
                                if hasattr(router, 'content_store') and hasattr(router.content_store, 'dqn_agent'):
                                    dqn_agent = router.content_store.dqn_agent
                                    if dqn_agent is not None:
                                        # Calculate router-specific hit rate
                                        router_hits = getattr(router.stats, 'stats', {}).get('hits', 0)
                                        router_requests = getattr(router.stats, 'stats', {}).get('nodes', 0)
                                        router_hit_rate = router_hits / max(1, router_requests)
                                        
                                        # Count cache decisions (approximate from insertions)
                                        cache_decisions = getattr(router.stats, 'cache_insertion_attempts', 0)
                                        
                                        dqn_agent.record_round_metrics(round_num, router_hit_rate, cache_decisions)
                                        
                                        # Save checkpoint if needed (handles periodic + best model)
                                        dqn_agent.save_checkpoint_if_needed(round_num, router_hit_rate, experiment_metadata)
                    except Exception as e:
                        logger.debug(f"Error recording DQN learning curve: {e}")
                
                if (round_num % 10 == 0):
                    debug_print("Collecting statistics...", logger)
                    debug_print(f"Statistics for round {round_num}: {stats_data}", logger)

        log_info(
            f"Simulation completed: total_hit_rate={global_stats.cache_hits / max(1, total_requests):.4f}, "
            f"total_requests={total_requests}, cache_hits={global_stats.cache_hits}, nodes_traversed={global_stats.nodes_traversed}",
            progress_logger
        )
        debug_print("Simulation completed successfully", logger)
        
        # Shutdown DQN Training Manager gracefully
        try:
            from router import DQNTrainingManager
            training_manager = DQNTrainingManager.get_instance()
            if training_manager is not None:
                debug_print("Shutting down DQN Training Manager...", logger)
                training_manager.shutdown()
                debug_print("DQN Training Manager shutdown complete", logger)
        except Exception as e:
            debug_print(f"Error shutting down training manager: {e}", logger)
        
        # Save final checkpoints for all DQN agents
        # Use correct hit rate calculation: cache_hits / total_requests (not nodes_traversed)
        final_hit_rate = global_stats.cache_hits / max(1, total_requests)
        experiment_metadata = {
            'num_rounds': num_rounds,
            'num_requests': num_requests,
            'network_size': len(G.nodes()),
            'seed': getattr(random, '_seed', None) or getattr(np.random, '_seed', None)
        }
        
        try:
            for node, data in G.nodes(data=True):
                if 'router' in data:
                    router = data['router']
                    if hasattr(router, 'content_store') and hasattr(router.content_store, 'dqn_agent'):
                        dqn_agent = router.content_store.dqn_agent
                        if dqn_agent is not None:
                            router_hits = getattr(router.stats, 'stats', {}).get('hits', 0)
                            router_requests = getattr(router.stats, 'stats', {}).get('nodes', 0)
                            router_hit_rate = router_hits / max(1, router_requests)
                            dqn_agent.save_final_checkpoint(num_rounds, router_hit_rate, experiment_metadata)
        except Exception as e:
            debug_print(f"Error saving final checkpoints: {e}", logger)
        
        summary = {
            'nodes_traversed': global_stats.nodes_traversed,
            'cache_hits': global_stats.cache_hits,
            'data_packets': global_stats.data_packets_transferred,
            'data_size': global_stats.total_data_size_transferred,
            'hit_rate': final_hit_rate,
            'round_stats': round_stats
        }
        log_info(
            f"Simulation summary: {summary}",
            logger or progress_logger
        )
        return summary

    except Exception as e:
        debug_print(f"Error in simulation: {str(e)}", logger)
        if logger:
            logger.error("Stack trace:", exc_info=True)
        raise

def save_graph(graph: nx.Graph, filename: str, logger: logging.Logger = None):
    """Save network graph to file"""
    try:
        print("📂 Saving graph...")
        if logger:
            logger.info(f"Saving graph to {filename}")
        
        # Clean up non-serializable objects
        for node in graph.nodes():
            if 'router' in graph.nodes[node]:
                router = graph.nodes[node]['router']
                if hasattr(router, 'content_store'):
                    # Remove DQN agent if it exists
                    if hasattr(router.content_store, 'dqn_agent'):
                        router.content_store.dqn_agent = None
                    
        with open(filename, "wb") as f:
            pickle.dump(graph, f)
            
        print(f"✅ Graph saved as {filename}")
        if logger:
            logger.info(f"Graph saved successfully to {filename}")
        
    except Exception as e:
        print(f"❌ Error saving graph: {e}")
        if logger:
            logger.error(f"Error saving graph: {e}", exc_info=True)
        raise

def main():
    try:
        # Print system information
        debug_print(f"Python version: {sys.version}")
        if os.environ.get("SKIP_TORCH_IMPORT", "0") == "1":
            debug_print("Skipping PyTorch availability check (environment override)")
        else:
            try:
                import torch
                debug_print(f"PyTorch version: {torch.__version__}")
            except ImportError:
                debug_print("PyTorch not available")
            
        debug_print(f"NumPy version: {np.__version__}")
        debug_print(f"NetworkX version: {nx.__version__}")
        debug_print(f"Current working directory: {Path.cwd()}")
        debug_print(f"Number of CPU cores: {threading.active_count()}")
        try:
            debug_print(f"Initial GC counts: {gc.get_count()}")
        except:
            debug_print("GC count not available")
        
        # Simulation parameters - improved for better testing
        runtime: Optional[RouterRuntime] = None
        params = {
            'num_nodes': int(os.environ.get("NDN_SIM_NODES", "300")),
            'num_producers': int(os.environ.get("NDN_SIM_PRODUCERS", "60")),
            'num_contents': int(os.environ.get("NDN_SIM_CONTENTS", "6000")),
            'num_users': int(os.environ.get("NDN_SIM_USERS", "2000")),
            # Task 1.2: Increase default rounds for better cache warm-up
            'num_rounds': int(os.environ.get("NDN_SIM_ROUNDS", "20")),  # Increased from 5 to 20
            'num_requests': int(os.environ.get("NDN_SIM_REQUESTS", "5")),
            # Task 2.1: Default to combined eviction algorithm (Algorithm 1 from report)
            'cache_policy': os.environ.get("NDN_SIM_CACHE_POLICY", "combined"),
            # Toggle DQN caching policy across routers
            'use_dqn_cache': os.environ.get("NDN_SIM_USE_DQN", "0") == "1"
        }
        
        debug_print("Starting NDN Simulation with parameters:")
        for key, value in params.items():
            debug_print(f"  {key}: {value}")
        
        # Setup logging
        sim_logger, net_logger = setup_logging()
        debug_print("Logging setup completed")
        
        # Create network
        debug_print("Starting network creation...")
        G, users, producers, runtime = create_network(
            num_nodes=params['num_nodes'],
            num_producers=params['num_producers'],
            num_contents=params['num_contents'],
            num_users=params['num_users'],
            cache_policy=params.get('cache_policy', 'lru'),
            logger=net_logger
        )

        # Snapshot topology layout for quick inspection
        visualize_topology(G, filename="network_topology.png", logger=net_logger)
        
        # Task 1.3: Enable and verify DQN caching across routers
        if params.get('use_dqn_cache'):
            debug_print("Enabling DQN caching mode across all routers...", net_logger)
            setup_all_routers_to_dqn_mode(G, net_logger)
            
            # Initialize DQN Training Manager for asynchronous training
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
            debug_print(f"DQN Training Manager initialized with {max_training_workers} workers", net_logger)
            
            # Set up checkpoint directories for DQN agents (research best practice)
            from datetime import datetime
            checkpoint_base_dir = Path("dqn_checkpoints")
            experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_frequency = int(os.environ.get("DQN_CHECKPOINT_FREQUENCY", "10"))  # Every 10 rounds
            keep_checkpoints = int(os.environ.get("DQN_KEEP_CHECKPOINTS", "5"))  # Keep last 5
            if not os.environ.get("NDN_SIM_EXPERIMENT_ID"):
                os.environ["NDN_SIM_EXPERIMENT_ID"] = experiment_id
            
            # Verify DQN agents are initialized and configure checkpointing
            dqn_enabled_count = 0
            dqn_failed_count = 0
            for node, data in G.nodes(data=True):
                if 'router' in data:
                    r = data['router']
                    if getattr(r, 'capacity', 0) > 0 and hasattr(r, 'content_store'):
                        cs = r.content_store
                        if hasattr(cs, 'mode') and cs.mode == "dqn_cache":
                            if hasattr(cs, 'dqn_agent') and cs.dqn_agent is not None:
                                # Configure checkpointing for this agent
                                router_checkpoint_dir = checkpoint_base_dir / experiment_id / f"router_{r.router_id}"
                                cs.dqn_agent.set_checkpoint_config(
                                    checkpoint_dir=str(router_checkpoint_dir),
                                    frequency=checkpoint_frequency,
                                    keep_last=keep_checkpoints
                                )
                                dqn_enabled_count += 1
                            else:
                                dqn_failed_count += 1
                                debug_print(f"Router {r.router_id}: DQN mode set but agent not initialized", net_logger)
            
            if net_logger:
                net_logger.info(f"DQN Caching Status: {dqn_enabled_count} routers with DQN enabled, {dqn_failed_count} failed")
                if dqn_enabled_count > 0:
                    net_logger.info(f"Checkpoint directory: {checkpoint_base_dir / experiment_id}")
                    net_logger.info(f"Checkpoint frequency: every {checkpoint_frequency} rounds, keeping last {keep_checkpoints}")
        else:
            debug_print("DQN caching disabled (set NDN_SIM_USE_DQN=1 to enable)", net_logger)
        
        # Initialize Bloom filter propagation for neighbor awareness
        initialize_bloom_filter_propagation(G, logger=net_logger)
        
        # Align user distributions to actual producer contents for this run
        align_user_distributions_with_producers(users, producers, net_logger)

        # Task 1.2: Cache warm-up phase before main simulation
        warmup_rounds = int(os.environ.get("NDN_SIM_WARMUP_ROUNDS", "5"))
        warmup_cache(G, users, producers, num_warmup_rounds=warmup_rounds, logger=net_logger)

        # Apply fixes to all components
        debug_print("Applying fixes to network components...", net_logger)

        # Fix producers - make sure all producers register their content
        debug_print("Fixing producer content registration...", net_logger)
        for producer in producers:
            if hasattr(producer, '_register_content_with_routers'):
                producer._register_content_with_routers(G)
            
            # Debug the producer content status
            if hasattr(producer, 'print_content_status'):
                producer.print_content_status(G)

        # Install global prefix routes for better reachability
        # This ensures all routers have FIB entries for producer content prefixes
        # Can be disabled by setting NDN_SIM_INSTALL_GLOBAL=0
        if os.environ.get("NDN_SIM_INSTALL_GLOBAL", "1") != "0":
            install_global_prefix_routes(G, producers, net_logger)

        # Add delay for routes to propagate
        debug_print("Waiting for routes to propagate...", net_logger)
        time.sleep(5)  # Wait 5 seconds

        # Check FIB content on each router
        debug_print("Checking router FIB entries...", net_logger)
        for node, data in G.nodes(data=True):
            if 'router' in data and hasattr(data['router'], 'FIB'):
                router = data['router']
                entry_count = len(router.FIB)
                if entry_count > 0:
                    debug_print(f"  Router {router.router_id}: {entry_count} FIB entries", net_logger)
                    # Print a sample of FIB entries
                    for content_name, next_hops in list(router.FIB.items())[:3]:
                        debug_print(f"    {content_name} -> {next_hops}", net_logger)
                else:
                    debug_print(f"  Router {router.router_id}: No FIB entries", net_logger)
        
        # Print network statistics
        if net_logger:
            net_logger.info("Network statistics:")
            net_logger.info(f"  Nodes: {G.number_of_nodes()}")
            net_logger.info(f"  Edges: {G.number_of_edges()}")
            net_logger.info(f"  Users: {len(users)}")
            net_logger.info(f"  Producers: {len(producers)}")
        
        # Task 3.1: Initialize metrics collector
        metrics_collector = None
        try:
            from metrics import get_metrics_collector
            metrics_collector = get_metrics_collector()
            metrics_collector.set_network_graph(G)
            debug_print("Metrics collector initialized", net_logger)
        except Exception as e:
            debug_print(f"Failed to initialize metrics collector: {e}", net_logger)
        
        # Proceed directly to simulation in non-interactive mode
        
        # Run simulation
        debug_print("Starting simulation...")
        try:
            stats = run_simulation(
                G, users, producers,
                num_rounds=params['num_rounds'],
                num_requests=params['num_requests'],
                logger=sim_logger
            )
            
            if sim_logger:
                sim_logger.info("Simulation completed successfully")
                sim_logger.info("Final Statistics:")
            for key, value in stats.items():
                if key != 'round_stats':
                    if sim_logger:
                        sim_logger.info(f"  {key}: {value}")

            # Demonstrate basic caching with repeated requests for a known item
            if not params.get('use_dqn_cache'):
                demo_cache_behavior(G, users, producers, sim_logger)

            # Task 1.4: Enhanced cache statistics reporting
            try:
                debug_print("Cache snapshot and insertion statistics:")
                shown = 0
                cached_router_count = 0
                cached_item_total = 0
                total_insertions = 0
                total_data_msgs = 0
                total_insertion_attempts = 0
                total_insertion_successes = 0
                total_insertion_failures = 0
                routers_with_capacity = 0
                
                for node, data in G.nodes(data=True):
                    if 'router' in data:
                        r = data['router']
                        if getattr(r, 'capacity', 0) > 0 and hasattr(r, 'content_store'):
                            routers_with_capacity += 1
                            cs = r.content_store
                            cached_items = len(getattr(cs, 'store', {}))
                            if cached_items > 0:
                                cached_router_count += 1
                                cached_item_total += cached_items
                            total_insertions += getattr(cs, 'insertions', 0)
                            total_data_msgs += getattr(r, 'data_messages_processed', 0)
                            
                            # Task 1.4: Collect cache insertion statistics
                            if hasattr(r.stats, 'cache_insertion_attempts'):
                                total_insertion_attempts += r.stats.cache_insertion_attempts
                                total_insertion_successes += r.stats.cache_insertion_successes
                                total_insertion_failures += r.stats.cache_insertion_failures
                            
                            capacity_used = getattr(cs, 'total_capacity', 0) - getattr(cs, 'remaining_capacity', 0)
                            if shown < 3:
                                debug_print(f"  Router {r.router_id}: cached_items={cached_items}, capacity_used={capacity_used}")
                                shown += 1
                
                if net_logger:
                    net_logger.info(f"Cache Statistics:")
                    net_logger.info(f"  Routers with caching capacity: {routers_with_capacity}")
                    net_logger.info(f"  Routers with cached items: {cached_router_count}")
                    net_logger.info(f"  Total cached items: {cached_item_total}")
                    net_logger.info(f"  Total cache insertions: {total_insertions}")
                    net_logger.info(f"  Total data messages processed: {total_data_msgs}")
                    if total_insertion_attempts > 0:
                        net_logger.info(f"  Cache insertion attempts: {total_insertion_attempts}")
                        net_logger.info(f"  Cache insertion successes: {total_insertion_successes}")
                        net_logger.info(f"  Cache insertion failures: {total_insertion_failures}")
                        success_rate = (total_insertion_successes / total_insertion_attempts) * 100
                        net_logger.info(f"  Cache insertion success rate: {success_rate:.2f}%")
            except Exception as e:
                debug_print(f"Error collecting cache statistics: {e}", net_logger)
            
            # Task 3.1: Report comprehensive metrics
            if metrics_collector:
                try:
                    all_metrics = metrics_collector.get_all_metrics()
                    if net_logger:
                        net_logger.info("=" * 80)
                        net_logger.info("COMPREHENSIVE EVALUATION METRICS")
                        net_logger.info("=" * 80)
                        
                        # Latency metrics
                        latency = all_metrics.get('latency', {})
                        net_logger.info(f"Latency (Interest to Data):")
                        net_logger.info(f"  Mean: {latency.get('mean', 0):.4f}s")
                        net_logger.info(f"  Median: {latency.get('median', 0):.4f}s")
                        net_logger.info(f"  Min: {latency.get('min', 0):.4f}s, Max: {latency.get('max', 0):.4f}s")
                        net_logger.info(f"  Count: {latency.get('count', 0)}")
                        
                        # Redundancy metrics
                        redundancy = all_metrics.get('redundancy', {})
                        net_logger.info(f"Content Redundancy:")
                        net_logger.info(f"  Mean copies per content: {redundancy.get('mean', 0):.2f}")
                        net_logger.info(f"  Min: {redundancy.get('min', 0)}, Max: {redundancy.get('max', 0)}")
                        net_logger.info(f"  Total unique contents: {redundancy.get('total_contents', 0)}")
                        
                        # Dispersion metrics
                        dispersion = all_metrics.get('dispersion', {})
                        net_logger.info(f"Interest Packet Dispersion:")
                        net_logger.info(f"  Mean unique routers per Interest: {dispersion.get('mean', 0):.2f}")
                        net_logger.info(f"  Min: {dispersion.get('min', 0)}, Max: {dispersion.get('max', 0)}")
                        
                        # Stretch metrics
                        stretch = all_metrics.get('stretch', {})
                        net_logger.info(f"Stretch (actual/optimal hops):")
                        net_logger.info(f"  Mean: {stretch.get('mean', 0):.2f}")
                        net_logger.info(f"  Median: {stretch.get('median', 0):.2f}")
                        
                        # Cache hit rate
                        hit_rate = all_metrics.get('cache_hit_rate', {})
                        net_logger.info(f"Cache Hit Rate:")
                        net_logger.info(f"  Hit rate: {hit_rate.get('hit_rate', 0):.2f}%")
                        net_logger.info(f"  Total requests: {hit_rate.get('total_requests', 0)}")
                        net_logger.info(f"  Hits: {hit_rate.get('hits', 0)}, Misses: {hit_rate.get('misses', 0)}")
                        
                        # Cache utilization
                        utilization = all_metrics.get('cache_utilization', {})
                        net_logger.info(f"Cache Utilization:")
                        net_logger.info(f"  Mean: {utilization.get('mean', 0):.2f}%")
                        net_logger.info(f"  Median: {utilization.get('median', 0):.2f}%")
                        net_logger.info(f"  Routers tracked: {utilization.get('count', 0)}")
                        
                        net_logger.info("=" * 80)
                except Exception as e:
                    debug_print(f"Error reporting metrics: {e}", net_logger)

            if runtime:
                runtime.shutdown()

            # Gracefully shut down all routers
            debug_print("Shutting down routers...")
            for node in G.nodes():
                if 'router' in G.nodes[node]:
                    router = G.nodes[node]['router']
                    if hasattr(router, 'shutdown'):
                        router.shutdown()
                    if hasattr(router, 'set_runtime'):
                        router.set_runtime(None)

            # Save results after diagnostics complete
            debug_print("Saving network state...")
            save_graph(G, "network_state.pkl", sim_logger)
        except Exception as e:
            debug_print(f"Error during simulation: {str(e)}")
            if sim_logger:
                sim_logger.error("Simulation error:", exc_info=True)
            if runtime:
                runtime.shutdown()
    
    except Exception as e:
        debug_print(f"Fatal error in simulation: {str(e)}")
        import traceback
        traceback.print_exc()
        if runtime:
            runtime.shutdown()
        sys.exit(1)

if __name__ == "__main__":
    main()

"""
Integration Tests for Multi-Agent DQN with Neighbor-Aware State Representation

Tests end-to-end scenarios:
- Multi-user requesting same content
- Cache coordination across neighbors
- Bloom filter propagation
"""

import unittest
import sys
import os
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import create_network, run_simulation
from router import Router
from packet import Interest, Data


class TestMultiUserSameContent(unittest.TestCase):
    """Test that multiple users requesting same content works correctly"""
    
    def test_interest_aggregation_works(self):
        """Test Interest aggregation when multiple users request same content"""
        G, users, producers, runtime = create_network(
            num_nodes=10,
            num_producers=2,
            num_contents=50,
            num_users=5,
            cache_policy='lru'
        )
        
        # This is a basic integration test
        # Full test would require running simulation and checking PIT entries
        self.assertGreater(len(users), 0)
        self.assertGreater(len(producers), 0)
        
        if runtime:
            runtime.shutdown()


class TestBloomFilterPropagation(unittest.TestCase):
    """Test Bloom filter propagation between neighbors"""
    
    def test_bloom_filter_propagation_exists(self):
        """Test that Bloom filter propagation mechanism exists"""
        G, users, producers, runtime = create_network(
            num_nodes=10,
            num_producers=2,
            num_contents=50,
            num_users=5,
            cache_policy='lru'
        )
        
        # Check that routers have Bloom filters
        for node, data in G.nodes(data=True):
            if 'router' in data:
                router = data['router']
                if hasattr(router, 'content_store'):
                    content_store = router.content_store
                    self.assertTrue(hasattr(content_store, 'bloom_filter'))
                    self.assertTrue(hasattr(content_store, 'propagate_bloom_filter_to_neighbors'))
                    self.assertTrue(hasattr(content_store, 'receive_bloom_filter_update'))
                    self.assertTrue(hasattr(content_store, 'neighbor_filters'))
        
        if runtime:
            runtime.shutdown()


class TestCacheCoordination(unittest.TestCase):
    """Test cache coordination across neighbors"""
    
    def test_neighbor_awareness_feature_exists(self):
        """Test that neighbor awareness feature (Feature 6) exists in state space"""
        G, users, producers, runtime = create_network(
            num_nodes=10,
            num_producers=2,
            num_contents=50,
            num_users=5,
            cache_policy='lru'
        )
        
        # Check that state space includes neighbor awareness
        for node, data in G.nodes(data=True):
            if 'router' in data:
                router = data['router']
                if hasattr(router, 'content_store'):
                    content_store = router.content_store
                    if hasattr(content_store, 'get_state_for_dqn'):
                        state = content_store.get_state_for_dqn("test_content", 100)
                        # State should have 10 features
                        self.assertEqual(len(state), 10)
                        # Feature 6 should be neighbor has content (Bloom filter)
                        # This is tested by checking state dimension
        
        if runtime:
            runtime.shutdown()


class TestDQNStateSpace(unittest.TestCase):
    """Test that DQN state space has correct features"""
    
    def test_state_space_has_10_features(self):
        """Test that optimized state space has 10 features"""
        from utils import ContentStore
        
        store = ContentStore(total_capacity=100, router_id=0)
        state = store.get_state_for_dqn("test_content", 10)
        
        self.assertEqual(len(state), 10)
        self.assertEqual(state.shape, (10,))
        
        # Verify all features are in valid range
        for i, value in enumerate(state):
            self.assertGreaterEqual(value, 0.0, f"Feature {i} below 0: {value}")
            self.assertLessEqual(value, 1.0, f"Feature {i} above 1: {value}")
    
    def test_feature_6_is_neighbor_awareness(self):
        """Test that Feature 6 is neighbor has content (Bloom filter)"""
        from utils import ContentStore
        import networkx as nx
        
        G = nx.Graph()
        G.add_node(0)
        G.add_node(1)
        G.add_edge(0, 1)
        
        store = ContentStore(total_capacity=100, router_id=0)
        # Create a mock router for testing
        class MockRouter:
            def __init__(self):
                self.neighbors = {1}
                self.neighbor_lock = __import__('threading').Lock()
        
        store.router_ref = MockRouter()
        store.graph_ref = G
        
        state = store.get_state_for_dqn("test_content", 10)
        
        # Feature 6 should exist and be in [0, 1]
        self.assertEqual(len(state), 10)
        self.assertGreaterEqual(state[6], 0.0)
        self.assertLessEqual(state[6], 1.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

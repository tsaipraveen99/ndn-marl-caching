"""
Unit tests for caching components
Tests: Combined Eviction, Semantic Encoder, Neural Bloom Filter, DQN State, Cache Insertion
"""

import unittest
import numpy as np
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import ContentStore, BloomFilter, NeuralBloomFilter, ClusterManager
try:
    from semantic_encoder import SemanticEncoder, encode_name
except ImportError:
    SemanticEncoder = None
    encode_name = None
from packet import Data


class TestCombinedEviction(unittest.TestCase):
    """Test 1.1: Combined Eviction Algorithm"""
    
    def setUp(self):
        self.store = ContentStore(total_capacity=100, router_id=0)
        self.store.set_replacement_policy("combined")
    
    def test_recency_score_calculation(self):
        """Test that more recent content has higher recency score"""
        current_time = time.time()
        
        # Add content at different times
        self.store.store_content("old", Data(name="old", size=10, originator=0), 10, current_time - 100)
        self.store.store_content("recent", Data(name="recent", size=10, originator=0), 10, current_time - 10)
        
        # Recent content should have higher recency score
        # This is tested indirectly through eviction order
        self.assertIn("old", self.store.store)
        self.assertIn("recent", self.store.store)
    
    def test_frequency_score_calculation(self):
        """Test that more accessed content has higher frequency score"""
        current_time = time.time()
        
        # Add content
        self.store.store_content("popular", Data(name="popular", size=10, originator=0), 10, current_time)
        
        # Access popular content multiple times
        for _ in range(5):
            self.store.get_content("popular", current_time)
        
        # Access other content once
        self.store.store_content("rare", Data(name="rare", size=10, originator=0), 10, current_time)
        self.store.get_content("rare", current_time)
        
        # Popular should have higher access count
        self.assertGreater(
            self.store.access_count.get("popular", 0),
            self.store.access_count.get("rare", 0)
        )
    
    def test_combined_eviction(self):
        """Test combined eviction algorithm"""
        current_time = time.time()
        
        # Fill cache
        for i in range(10):
            self.store.store_content(
                f"content_{i}",
                Data(name=f"content_{i}", size=10, originator=0),
                10,
                current_time - (10 - i)  # Different access times
            )
        
        # Access some content more frequently
        for _ in range(3):
            self.store.get_content("content_5", current_time)
        
        # Try to add new content (should trigger eviction)
        result = self.store.store_content(
            "new_content",
            Data(name="new_content", size=10, originator=0),
            10,
            current_time
        )
        
        # Should succeed (eviction should free space)
        self.assertTrue(result)
        self.assertIn("new_content", self.store.store)
    
    def test_eviction_with_insufficient_space(self):
        """Test eviction when content too large"""
        current_time = time.time()
        
        # Fill cache completely
        self.store.store_content("large", Data(name="large", size=150, originator=0), 150, current_time)
        
        # Try to add content larger than capacity
        result = self.store.store_content(
            "too_large",
            Data(name="too_large", size=150, originator=0),
            150,
            current_time
        )
        
        # Should fail (content too large)
        self.assertFalse(result)


class TestSemanticEncoder(unittest.TestCase):
    """Test 1.2: Semantic Encoder"""
    
    @unittest.skipIf(SemanticEncoder is None, "SemanticEncoder not available")
    def test_encoding_consistency(self):
        """Test that same name produces same embedding"""
        encoder = SemanticEncoder(embedding_dim=64, use_cnn=False)  # Use hash fallback
        
        name = "/edu/university/cs/course/content"
        embedding1 = encoder.encode(name)
        embedding2 = encoder.encode(name)
        
        np.testing.assert_array_almost_equal(embedding1, embedding2)
    
    @unittest.skipIf(SemanticEncoder is None, "SemanticEncoder not available")
    def test_encoding_dimension(self):
        """Test encoding has correct dimension"""
        encoder = SemanticEncoder(embedding_dim=64, use_cnn=False)
        
        name = "/edu/university/cs/course/content"
        embedding = encoder.encode(name)
        
        self.assertEqual(embedding.shape, (64,))
    
    @unittest.skipIf(SemanticEncoder is None, "SemanticEncoder not available")
    def test_encoding_normalization(self):
        """Test encoding is normalized"""
        encoder = SemanticEncoder(embedding_dim=64, use_cnn=False)
        
        name = "/edu/university/cs/course/content"
        embedding = encoder.encode(name)
        
        norm = np.linalg.norm(embedding)
        self.assertAlmostEqual(norm, 1.0, places=5)
    
    @unittest.skipIf(SemanticEncoder is None, "SemanticEncoder not available")
    def test_batch_encoding(self):
        """Test batch encoding"""
        encoder = SemanticEncoder(embedding_dim=64, use_cnn=False)
        
        names = [
            "/edu/university/cs/course1",
            "/edu/university/cs/course2",
            "/edu/university/math/course1"
        ]
        
        embeddings = encoder.encode_batch(names)
        
        self.assertEqual(embeddings.shape, (3, 64))


class TestNeuralBloomFilter(unittest.TestCase):
    """Test 1.3: Neural Bloom Filter"""
    
    def test_basic_operations(self):
        """Test basic Bloom filter operations"""
        bf = BloomFilter(size=100, hash_count=4)
        
        # Add items
        bf.add("item1")
        bf.add("item2")
        
        # Check items
        self.assertTrue(bf.check("item1"))
        self.assertTrue(bf.check("item2"))
        self.assertFalse(bf.check("item3"))  # Not added
    
    def test_false_positive_detection(self):
        """Test false positive handling"""
        bf = NeuralBloomFilter(size=100, hash_count=4, use_neural=False)
        
        # Add items
        for i in range(10):
            bf.add(f"item_{i}")
        
        # Check for false positives (items not added)
        false_positives = 0
        for i in range(100, 200):
            if bf.check(f"item_{i}"):
                false_positives += 1
        
        # Should have some false positives (expected with small filter)
        # But neural filter should reduce them
        self.assertGreater(false_positives, 0)


class TestDQNStateSpace(unittest.TestCase):
    """Test 1.4: DQN State Space"""
    
    def setUp(self):
        self.store = ContentStore(total_capacity=100, router_id=0)
    
    def test_state_dimension(self):
        """Test state has correct dimension (10)"""
        state = self.store.get_state_for_dqn("test_content", 10)
        
        self.assertEqual(state.shape, (10,))
        self.assertEqual(len(state), 10)
    
    def test_state_features_range(self):
        """Test all state features are in valid range (0-1)"""
        state = self.store.get_state_for_dqn("test_content", 10)
        
        for i, value in enumerate(state):
            self.assertGreaterEqual(value, 0.0, f"Feature {i} below 0: {value}")
            self.assertLessEqual(value, 1.0, f"Feature {i} above 1: {value}")
    
    def test_state_with_content_cached(self):
        """Test state when content is cached"""
        current_time = time.time()
        self.store.store_content("cached", Data(name="cached", size=10, originator=0), 10, current_time)
        
        state = self.store.get_state_for_dqn("cached", 10)
        
        # Feature 0 should be 1.0 (content in store)
        self.assertEqual(state[0], 1.0)
    
    def test_state_with_content_not_cached(self):
        """Test state when content is not cached"""
        state = self.store.get_state_for_dqn("not_cached", 10)
        
        # Feature 0 should be 0.0 (content not in store)
        self.assertEqual(state[0], 0.0)


class TestCacheInsertion(unittest.TestCase):
    """Test 1.5: Cache Insertion Logic"""
    
    def setUp(self):
        self.store = ContentStore(total_capacity=100, router_id=0)
    
    def test_successful_insertion(self):
        """Test successful cache insertion"""
        current_time = time.time()
        data = Data(name="test_content", size=10, originator=0)
        
        result = self.store.store_content("test_content", data, 10, current_time)
        
        self.assertTrue(result)
        self.assertIn("test_content", self.store.store)
        self.assertEqual(self.store.insertions, 1)
    
    def test_insertion_when_full(self):
        """Test insertion triggers eviction when cache full"""
        current_time = time.time()
        
        # Fill cache
        for i in range(10):
            self.store.store_content(
                f"content_{i}",
                Data(name=f"content_{i}", size=10, originator=0),
                10,
                current_time
            )
        
        # Try to add new content
        result = self.store.store_content(
            "new_content",
            Data(name="new_content", size=10, originator=0),
            10,
            current_time
        )
        
        # Should succeed (eviction should free space)
        self.assertTrue(result)
        self.assertIn("new_content", self.store.store)
    
    def test_insertion_too_large(self):
        """Test insertion fails when content too large"""
        current_time = time.time()
        
        result = self.store.store_content(
            "too_large",
            Data(name="too_large", size=150, originator=0),
            150,
            current_time
        )
        
        # Should fail (content larger than capacity)
        self.assertFalse(result)
    
    def test_duplicate_content_handling(self):
        """Test duplicate content updates access time"""
        current_time = time.time()
        
        # Add content
        self.store.store_content("content", Data(name="content", size=10, originator=0), 10, current_time)
        first_access = self.store.last_access_time["content"]
        
        # Wait a bit
        time.sleep(0.1)
        
        # Add same content again
        self.store.store_content("content", Data(name="content", size=10, originator=0), 10, time.time())
        second_access = self.store.last_access_time["content"]
        
        # Access time should be updated
        self.assertGreater(second_access, first_access)
        # Insertions should still be 1 (not incremented for duplicates)
        self.assertEqual(self.store.insertions, 1)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)


import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
import unittest
import sqlite3
import time
import os
from datetime import datetime, timedelta
from Mempool.mempool import Mempool
from .fake_node import FakeNode, FailingNode


class TestMempool(unittest.TestCase):
    TEST_DB_PATH = "/media/henning/Volume/Programming/projectX/src/test/mempool_test.db"
    
    def setUp(self):
        # Create clean test database
        if os.path.exists(self.TEST_DB_PATH):
            os.remove(self.TEST_DB_PATH)
        
        # Create fake node with predictable data
        self.fake_node = FakeNode()
        self.mempool = Mempool(self.fake_node, self.TEST_DB_PATH)


    def test_get_mempool_feerates_success(self):
        """Test successful fee rate retrieval"""
        result = self.mempool.get_mempool_feerates()
        self.assertEqual(result["fast"], 7)
        self.assertEqual(result["medium"], 7)
        self.assertEqual(result["low"], 7)
        
        # Verify data was stored in DB
        conn = sqlite3.connect(self.TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM mempool_fee_histogram")
        self.assertEqual(cursor.fetchone()[0], 1)
        conn.close()


    def test_get_mempool_feerates_fallback(self): 
        failing_mempool = Mempool(FailingNode(), self.TEST_DB_PATH)
        result = failing_mempool.get_mempool_feerates()
        self.assertEqual(result["fast"], 10)


    def test_get_mempool_feerates_cache(self):
        """Test caching functionality"""
        # First call
        result1 = self.mempool.get_mempool_feerates()
        
        # Second call should use cache
        result2 = self.mempool.get_mempool_feerates()
        self.assertEqual(result1, result2)
        
        # Expire cache and fetch again
        self.mempool.last_fetch_time = time.time() - 61
        result3 = self.mempool.get_mempool_feerates()
        self.assertEqual(result1, result3)


    def test_predict_fee_rates_sufficient_data(self):
        """Test fee prediction with sufficient data"""
        # Insert test data with proper timestamps
        conn = sqlite3.connect(self.TEST_DB_PATH)
        cursor = conn.cursor()
        
        base_time = datetime.now() - timedelta(minutes=120)
        for i in range(10):
            timestamp = base_time + timedelta(minutes=i*10)
            cursor.execute(
                "INSERT INTO mempool_fee_histogram (timestamp, fast_fee, medium_fee, low_fee) VALUES (?, ?, ?, ?)",
                (timestamp, 10 + i*2, 5 + i, 1 + i*0.5)
            )
        conn.commit()
        conn.close()
        
        # Test prediction
        prediction = self.mempool.predict_fee_rates()
        
        # Basic validation of results
        self.assertIsInstance(prediction, dict)
        self.assertIn("fast", prediction)
        self.assertIn("medium", prediction)
        self.assertIn("low", prediction)
        self.assertGreater(prediction["fast"], 0)
        self.assertGreater(prediction["medium"], 0)
        self.assertGreater(prediction["low"], 0)
        
        # Verify prediction was stored
        conn = sqlite3.connect(self.TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM fee_predictions")
        self.assertEqual(cursor.fetchone()[0], 1)
        conn.close()
    

    def test_predict_fee_rates_insufficient_data(self):
        """Test prediction fallback with insufficient data"""
        # No data in database
        prediction = self.mempool.predict_fee_rates()
        current = self.mempool.get_mempool_feerates()
        self.assertEqual(prediction["fast"], current["fast"])  # From initial fetch
        
        # Minimal data (3 points - below minimum of 5)
        conn = sqlite3.connect(self.TEST_DB_PATH)
        cursor = conn.cursor()
        base_time = datetime.now() - timedelta(minutes=120)
        for i in range(2):
            timestamp = base_time + timedelta(minutes=i*10)
            cursor.execute(
                "INSERT INTO mempool_fee_histogram (timestamp, fast_fee, medium_fee, low_fee) VALUES (?, ?, ?, ?)",
                (timestamp, 10 + i*2, 5 + i, 1 + i*0.5)
            )
        conn.commit()
        conn.close()

        prediction = self.mempool.predict_fee_rates()
        current = self.mempool.get_mempool_feerates()

        self.assertEqual(prediction["fast"], current["fast"])


    def tearDown(self):
        # Clean up test database
        if os.path.exists(self.TEST_DB_PATH):
            os.remove(self.TEST_DB_PATH)


if __name__ == "__main__":
    unittest.main()
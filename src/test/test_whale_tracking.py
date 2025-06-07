import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
import unittest
import sqlite3
import os
import json
from datetime import datetime, timedelta
from WhaleTracking.whale_tracking import WhaleTracking
from fake_node import FakeNode


class TestWhaleTracking(unittest.TestCase):
    TEST_DB_PATH = "/media/henning/Volume/Programming/projectX/src/test/whale_test.db"
    
    def setUp(self):
        # Create clean test database
        if os.path.exists(self.TEST_DB_PATH):
            os.remove(self.TEST_DB_PATH)
        
        # Create fake node with test data
        self.fake_node = FakeNode()
        self.whale_tracker = WhaleTracking(self.fake_node, self.TEST_DB_PATH)
        
        # Setup test transactions and balances
        self.setup_test_data()

    def setup_test_data(self):
        """Create test transactions and balances"""
        # Whale address with high balance
        self.whale_address = "whale_address_1"
        self.fake_node.set_balance(self.whale_address, 5000)  # 5000 BTC

        # Regular address
        self.regular_address = "regular_address_1"
        self.fake_node.set_balance(self.regular_address, 100)
        
        # Create test transaction
        self.txid1 = "txid_001"
        self.fake_node.add_test_transaction(
            txid=self.txid1,
            inputs=[{"txid": "prev_tx_001", "vout": 0, "value": 50.0, "address": self.whale_address}],
            outputs=[
                {"value": 20.0, "scriptPubKey": {"address": "output_address1"}},
                {"value": 29.5, "scriptPubKey": {"address": "output_address2"}}
            ],
            total_sent=49.5
        )
        
        # Set mempool transactions
        self.fake_node.set_mempool_txids([self.txid1])


    def insert_test_transactions(self):
        """Insert test transactions directly into database"""
        conn = sqlite3.connect(self.TEST_DB_PATH)
        cursor = conn.cursor()
        
        # Insert whale transactions
        base_time = datetime.now() - timedelta(days=7)
        for i in range(10):
            timestamp = base_time + timedelta(hours=i)
            cursor.execute(
                """INSERT INTO whale_transactions 
                (txid, timestamp, total_sent, fee_per_vbyte)
                VALUES (?, ?, ?, ?)""",
                (f"test_tx_{i}", timestamp, 50 + i*10, 5 + i*0.5)
            )
            cursor.execute(
                "INSERT INTO transaction_inputs (txid, address) VALUES (?, ?)",
                (f"test_tx_{i}", self.whale_address)
            )
        
        conn.commit()
        conn.close()

    
    def test_process_transaction(self):
        """Test processing a single transaction"""
        # Process whale transaction
        result = self.whale_tracker.process_transaction(self.txid1, 10, 50000)
        self.assertTrue(result)
        
        # Verify transaction was stored
        conn = sqlite3.connect(self.TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM whale_transactions WHERE txid = ?", (self.txid1,))
        self.assertEqual(cursor.fetchone()[0], 1)
        
        # Verify inputs were stored
        cursor.execute("SELECT COUNT(*) FROM transaction_inputs WHERE txid = ?", (self.txid1,))
        self.assertEqual(cursor.fetchone()[0], 1)
        
        # Verify outputs were stored
        cursor.execute("SELECT COUNT(*) FROM transaction_outputs WHERE txid = ?", (self.txid1,))
        self.assertEqual(cursor.fetchone()[0], 2)
        conn.close()

    
    def test_process_mempool_transactions(self):
        """Test processing mempool transactions"""
        processed = self.whale_tracker.process_mempool_transactions(threshold=30)
        self.assertEqual(processed, 1)  # Only one transaction above threshold
        
        # Verify only whale transaction was processed
        conn = sqlite3.connect(self.TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT txid FROM whale_transactions")
        self.assertEqual(cursor.fetchone()[0], self.txid1)
        conn.close()

    
    def test_analyze_whale_behavior(self):
        """Test whale behavior analysis"""
        # Insert test data with LARGE transactions
        conn = sqlite3.connect(self.TEST_DB_PATH)
        cursor = conn.cursor()
        
        base_time = datetime.now() - timedelta(days=7)
        # Create transactions with amounts > 100 BTC
        for i in range(10):
            timestamp = base_time + timedelta(hours=i)
            # Large amounts: 150, 200, 250, ... 600 BTC
            total_sent = 150 + i * 50
            cursor.execute(
                """INSERT INTO whale_transactions 
                (txid, timestamp, total_sent, fee_per_vbyte)
                VALUES (?, ?, ?, ?)""",
                (f"test_tx_{i}", timestamp, total_sent, 5 + i*0.5)
            )
            cursor.execute(
                "INSERT INTO transaction_inputs (txid, address) VALUES (?, ?)",
                (f"test_tx_{i}", self.whale_address)
            )
        
        conn.commit()
        conn.close()
        
        # Analyze behavior
        behavior = self.whale_tracker.analyze_whale_behavior(self.whale_address)
        
        # Verify classification - should be "Large Transactor"
        self.assertEqual(behavior, "Large Transactor")
        
        # Verify behavior was stored
        conn = sqlite3.connect(self.TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT behavior_pattern FROM whale_behavior WHERE address = ?", (self.whale_address,))
        self.assertEqual(cursor.fetchone()[0], "Large Transactor")
        conn.close()

    
    def test_analyze_insufficient_data(self):
        """Test behavior analysis with insufficient data"""
        behavior = self.whale_tracker.analyze_whale_behavior("unknown_address")
        self.assertEqual(behavior, "Insufficient Data")

    
    def test_get_whale_addresses(self):
        """Test whale address identification"""
        # Insert test transactions
        self.insert_test_transactions()
        
        # Process whale transaction to create inputs
        self.whale_tracker.process_transaction(self.txid1, 10, 50000)
        
        # Get whale addresses
        whale_addresses = self.whale_tracker.get_whale_addresses(min_balance=1000)
        
        # Verify whale address was identified
        self.assertIn(self.whale_address, whale_addresses)
        self.assertNotIn(self.regular_address, whale_addresses)

    
    def test_track_whale_balances(self):
        """Test tracking whale balances"""
        # Process whale transaction to ensure address is in the system
        btc_price = 50000.0
        self.whale_tracker.process_transaction(self.txid1, 10, btc_price)
        
        # Track balance directly (bypassing identification for this test)
        self.whale_tracker.track_whale_balances([self.whale_address])
        
        # Verify balance was stored
        conn = sqlite3.connect(self.TEST_DB_PATH)
        cursor = conn.cursor()
        
        # Check if record exists
        cursor.execute("SELECT COUNT(*) FROM whale_balance_history WHERE address = ?", 
                      (self.whale_address,))
        count = cursor.fetchone()[0]
        self.assertEqual(count, 1, "Balance record not found")
        
        # Check balance value
        cursor.execute("SELECT confirmed_balance FROM whale_balance_history WHERE address = ?", 
                      (self.whale_address,))
        balance = cursor.fetchone()[0]
        self.assertEqual(balance, 5000.0, "Balance value incorrect")
        
        conn.close()

    
    def tearDown(self):
        # Clean up test database
        if os.path.exists(self.TEST_DB_PATH):
            os.remove(self.TEST_DB_PATH)
    


if __name__ == "__main__":
    unittest.main()
   
    #TEST_DB_PATH = "/media/henning/Volume/Programming/projectX/src/test/whale_test.db"
    #fake_node = FakeNode()
    #whale_tracker = WhaleTracking(fake_node, TEST_DB_PATH)
    #whale_tracker.analyze_whale_behavior("whale_address_1")    
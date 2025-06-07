import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
import unittest
import sqlite3
import os
from WhaleTracking.whale_alert import WhaleAlerts
from fake_alert_sender import FakeAlertSender

class TestWhaleAlerts(unittest.TestCase):
    TEST_DB_PATH = "/media/henning/Volume/Programming/projectX/src/test/alerts_test.db"
    
    def setUp(self):
        # Create clean test database
        if os.path.exists(self.TEST_DB_PATH):
            os.remove(self.TEST_DB_PATH)
        
        # Create fake alert sender
        self.fake_sender = FakeAlertSender()
        
        # Initialize WhaleAlerts
        self.whale_alerts = WhaleAlerts(self.TEST_DB_PATH, self.fake_sender)


    def create_test_tx(self, txid, amount_btc, amount_usd, fee_rate, tx_out_addr):
        """Create a test transaction dictionary"""
        return {
            "txid": txid,
            "sum_btc_sent": amount_btc,
            "sum_usd_sent": amount_usd,
            "fee_per_vbyte": fee_rate,
            "tx_in_addr": ["input_addr_1", "input_addr_2"],
            "tx_out_addr": tx_out_addr
        }

    
    def test_detect_whale_move(self):
        """Test detection of standard whale move"""
        # Create whale move transaction
        tx = self.create_test_tx("txid_001", 20.0, 1500000.0, 50.0, ["output_addr_1", "output_addr_2"])
        
        # Detect activity
        self.whale_alerts.detect_unusual_activity(tx)
        
        # Verify alert was sent
        self.assertEqual(len(self.fake_sender.sent_alerts), 1)
        self.assertIn("Whale Move Alert", self.fake_sender.sent_alerts[0])
        
        # Verify alert was stored
        conn = sqlite3.connect(self.TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM alerted_events")
        self.assertEqual(cursor.fetchone()[0], 1)
        cursor.execute("SELECT alert_type FROM alert_history WHERE txid = ?", ("txid_001",))
        self.assertEqual(cursor.fetchone()[0], "WHALE_MOVE")
        conn.close()
    

    def test_detect_miner_bribe(self):
        """Test detection of miner bribe"""
        # Create miner bribe transaction
        tx = self.create_test_tx("txid_002", 5.0, 5000000.0, 1500.0, ["output_addr_1"])
        
        # Detect activity
        self.whale_alerts.detect_unusual_activity(tx)
        
        # Verify alert was sent
        self.assertEqual(len(self.fake_sender.sent_alerts), 1)
        self.assertIn("Miner Bribe Alert", self.fake_sender.sent_alerts[0])
        
        # Verify alert type
        conn = sqlite3.connect(self.TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT alert_type FROM alert_history WHERE txid = ?", ("txid_002",))
        self.assertEqual(cursor.fetchone()[0], "MINER_BRIBE")
        conn.close()

    
    def test_detect_exchange_flow(self):
        """Test detection of exchange flow"""
        # Create exchange flow transaction
        tx = self.create_test_tx("txid_003", 50.0, 3000000.0, 100.0, 
                                ["bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh", "output_addr_2"])
        
        # Detect activity
        self.whale_alerts.detect_unusual_activity(tx)
        
        # Verify alert was sent
        self.assertEqual(len(self.fake_sender.sent_alerts), 1)
        self.assertIn("Exchange Flow Alert", self.fake_sender.sent_alerts[0])
        
        # Verify alert type
        conn = sqlite3.connect(self.TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT alert_type FROM alert_history WHERE txid = ?", ("txid_003",))
        self.assertEqual(cursor.fetchone()[0], "EXCHANGE_FLOW")
        conn.close()


    def test_duplicate_alerts(self):
        """Test that duplicate alerts are not sent"""
        # Create transaction
        tx = self.create_test_tx("txid_004", 30.0, 2000000.0, 100.0, ["output_addr_1"])
        
        # First detection
        self.whale_alerts.detect_unusual_activity(tx)
        
        # Second detection
        self.whale_alerts.detect_unusual_activity(tx)
        
        # Should only have one alert
        self.assertEqual(len(self.fake_sender.sent_alerts), 1)
        
        # Should only have one record in alerted_events
        conn = sqlite3.connect(self.TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM alerted_events")
        self.assertEqual(cursor.fetchone()[0], 1)
        conn.close()


    def test_below_threshold(self):
        """Test transactions below threshold are not alerted"""
        # Create below-threshold transaction
        tx = self.create_test_tx("txid_005", 5.0, 500000.0, 100.0, ["output_addr_1"])
        
        # Detect activity
        self.whale_alerts.detect_unusual_activity(tx)
        
        # Verify no alert was sent
        self.assertEqual(len(self.fake_sender.sent_alerts), 0)
        
        # Verify no records in database
        conn = sqlite3.connect(self.TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM alerted_events")
        self.assertEqual(cursor.fetchone()[0], 0)
        cursor.execute("SELECT COUNT(*) FROM alert_history")
        self.assertEqual(cursor.fetchone()[0], 0)
        conn.close()


    def test_invalid_data(self):
        """Test handling of invalid transaction data"""
        # Missing required fields
        tx = {"txid": "txid_006", "sum_btc_sent": 10.0}
        self.whale_alerts.detect_unusual_activity(tx)
        
        # Wrong data type
        self.whale_alerts.detect_unusual_activity("invalid data")
        
        # Empty data
        self.whale_alerts.detect_unusual_activity({})
        
        # Verify no alerts were sent
        self.assertEqual(len(self.fake_sender.sent_alerts), 0)


    def test_alert_history_storage(self):
        """Test that alert history is stored correctly"""
        # Create transaction
        tx = self.create_test_tx("txid_007", 25.0, 2500000.0, 100.0, ["output_addr_1"])
        
        # Detect activity
        self.whale_alerts.detect_unusual_activity(tx)
        
        # Verify history storage
        conn = sqlite3.connect(self.TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM alert_history WHERE txid = ?", ("txid_007",))
        row = cursor.fetchone()
        
        self.assertIsNotNone(row)
        self.assertEqual(row[2], "txid_007")
        self.assertEqual(row[3], "WHALE_MOVE")
        self.assertEqual(row[4], 25.0)
        self.assertEqual(row[5], 2500000.0)
        self.assertEqual(row[6], 100.0)
        
        conn.close()


    def test_event_hashing(self):
        """Test consistent event hashing"""
        tx1 = self.create_test_tx("txid_008", 20.0, 1500000.0, 50.0, ["addr1"])
        tx2 = self.create_test_tx("txid_008", 20.0, 1500000.0, 50.0, ["addr1"])
        
        hash1 = self.whale_alerts._generate_event_hash(tx1)
        hash2 = self.whale_alerts._generate_event_hash(tx2)
        
        # Same transaction should have same hash
        self.assertEqual(hash1, hash2)
        
        # Different transaction should have different hash
        tx3 = self.create_test_tx("txid_009", 20.0, 1500000.0, 50.0, ["addr1"])
        hash3 = self.whale_alerts._generate_event_hash(tx3)
        self.assertNotEqual(hash1, hash3)


    def tearDown(self):
        # Clean up test database
        if os.path.exists(self.TEST_DB_PATH):
            os.remove(self.TEST_DB_PATH)


if __name__ == "__main__":
    unittest.main()
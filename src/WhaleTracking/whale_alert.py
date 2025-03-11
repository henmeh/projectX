import sqlite3
import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
from Helper.helperfunctions import create_table, send_telegram_alert
import datetime

class WhaleAlerts:
    def __init__(self, db_path: str = "/media/henning/Volume/Programming/projectX/src/mempool_transactions.db", alert_threshold=1000):
        self.alert_threshold = alert_threshold
        self.db_path = db_path
        try:
            create_table(self.db_path, """CREATE TABLE IF NOT EXISTS alerted_transactions (txid TEXT PRIMARY KEY, timestamp TEXT)""")
        except Exception as e:
            print(f"❌ Database creation filed: {e}")
    

    def get_alert_threshold(self) -> int:
        """
        Returns the threshold for the whale alert
        """
        return self.alert_threshold
    
    
    def is_alerted(self, txid):
        """Check if a transaction has already been alerted."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM alerted_transactions WHERE txid = ?", (txid,))
            return cursor.fetchone() is not None


    def mark_as_alerted(self, txid, timestamp):
        """Mark a transaction as alerted."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR IGNORE INTO alerted_transactions (txid, timestamp) VALUES (?, ?)", (txid, timestamp))
            conn.commit()


    def detect_unusual_activity(self, tx: dict):
        """
        Identify unusually large transactions and send alerts only once.
        """
        
        # Skip if this transaction has already been alerted
        if self.is_alerted(tx["txid"]) == False:
                
            message = (
                        f"🚨 *Whale Alert!* 🚨\n"
                        f"💰 *{tx['sum_btc_sent']} BTC* transferred!\n"
                        f"📥 *From:* {', '.join(tx['tx_in_addr'][:3])}...\n"
                        f"📤 *To:* {', '.join(tx['tx_out_addr'][:3])}...\n"
                        f"⏳ *Time:* {datetime.datetime.now()}\n"
                        f"🔗 [View Transaction](https://mempool.space/tx/{tx['txid']})"
                    )

            # Send alert and mark it as alerted
            send_telegram_alert(message)
            self.mark_as_alerted(tx["txid"], datetime.datetime.now())
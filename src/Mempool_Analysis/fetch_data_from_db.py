import sqlite3
import json
from datetime import datetime, timedelta


class FetchData():
    def __init__(self, days=7):
        self.db_mempool_transactions_path = "/media/henning/Volume/Programming/projectX/src/mempool_transactions.db"
        self.days = days

    def fetch_whale_transactions(self):
        """
        Fetches whale transactions from the last `days` days.
        Parses JSON-encoded addresses and returns structured data.
        """
        conn = sqlite3.connect(self.db_mempool_transactions_path)
        cursor = conn.cursor()

        # Define the date range
        start_date = datetime.now() - timedelta(days=self.days)

        # SQL Query to fetch transactions within the given time frame
        query = """
        SELECT timestamp, txid, tx_in_addr, tx_out_addr, total_sent, fee_paid, fee_per_vbyte
        FROM mempool_transactions
        WHERE timestamp >= ?
        """
        cursor.execute(query, (start_date.strftime("%Y-%m-%d %H:%M:%S"),))
        
        transactions = []
        
        for row in cursor.fetchall():
            timestamp, txid, tx_in_addr, tx_out_addr, total_sent, fee_paid, fee_per_vbyte = row
            
            # Parse JSON-encoded addresses
            try:
                tx_in_addr = json.loads(tx_in_addr) if tx_in_addr else []
                tx_out_addr = json.loads(tx_out_addr) if tx_out_addr else []
            except json.JSONDecodeError:
                tx_in_addr, tx_out_addr = [], []

            transactions.append({
                "timestamp": timestamp,
                "txid": txid,
                "tx_in_addr": tx_in_addr,
                "tx_out_addr": tx_out_addr,
                "total_sent": total_sent,
                "fee_paid": fee_paid,
                "fee_per_vbyte": fee_per_vbyte
            })

        conn.close()
        return transactions
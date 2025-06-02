import numpy as np
from sklearn.ensemble import IsolationForest
from Helper.helperfunctions import create_table, fetch_btc_price, store_data, fetch_whale_transactions

class WhaleTracking:
    def __init__(self, node, db_path: str, days=7):
        self.node = node
        self.db_path = db_path
        self.days = days
        
        # Create optimized tables
        create_table(db_path, '''CREATE TABLE IF NOT EXISTS whale_transactions (
            txid TEXT PRIMARY KEY,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            size INTEGER,
            vsize INTEGER,
            weight INTEGER,
            fee_paid REAL,
            fee_per_vbyte REAL,
            total_sent REAL,
            btcusd REAL)''')
            
        create_table(db_path, '''CREATE TABLE IF NOT EXISTS transaction_inputs (
            txid TEXT,
            address TEXT,
            value REAL,
            FOREIGN KEY(txid) REFERENCES whale_transactions(txid))''')
            
        create_table(db_path, '''CREATE TABLE IF NOT EXISTS transaction_outputs (
            txid TEXT,
            address TEXT,
            value REAL,
            FOREIGN KEY(txid) REFERENCES whale_transactions(txid))''')
            
        create_table(db_path, '''CREATE TABLE IF NOT EXISTS whale_balance_history (
            address TEXT,
            timestamp DATETIME,
            confirmed_balance REAL,
            unconfirmed_balance REAL,
            PRIMARY KEY (address, timestamp))''')
            
        create_table(db_path, '''CREATE TABLE IF NOT EXISTS whale_behavior (
            address TEXT PRIMARY KEY,
            behavior_pattern TEXT,
            last_updated DATETIME)''')

    async def process_tx_batch(self, txids: list, threshold: float):
        """Async batch processing of transactions"""
        btc_price = fetch_btc_price()
        whale_alert = WhaleAlerts(self.db_path)
        
        async with aiohttp.ClientSession() as session:
            tasks = [self._process_single_tx(session, txid, threshold, btc_price, whale_alert) 
                    for txid in txids]
            await asyncio.gather(*tasks)

    async def _process_single_tx(self, session, txid, threshold, btc_price, whale_alert):
        """Process a single transaction asynchronously"""
        try:
            tx = await self.node.async_rpc_call("getrawtransaction", [txid, True])
            
            if "error" in tx:
                return
                
            tx = tx["result"]
            total_sent = sum(out["value"] for out in tx.get("vout", []))
            
            if total_sent < threshold:
                return
                
            # Process inputs
            input_sum = 0
            input_addresses = []
            for vin in tx.get("vin", []):
                if "txid" in vin:
                    prev_tx = await self.node.async_rpc_call("getrawtransaction", [vin["txid"], True])
                    if "error" not in prev_tx:
                        prev_out = prev_tx["result"]["vout"][vin["vout"]]
                        input_sum += prev_out["value"]
                        addr = prev_out["scriptPubKey"].get("address", "")
                        if addr:
                            input_addresses.append(addr)
                            store_data(
                                self.db_path,
                                "INSERT INTO transaction_inputs (txid, address, value) VALUES (?, ?, ?)",
                                (txid, addr, prev_out["value"])
                            )
            
            # Process outputs
            output_addresses = []
            for vout in tx.get("vout", []):
                addr = vout["scriptPubKey"].get("address", "")
                if addr:
                    output_addresses.append(addr)
                    store_data(
                        self.db_path,
                        "INSERT INTO transaction_outputs (txid, address, value) VALUES (?, ?, ?)",
                        (txid, addr, vout["value"])
                    )
            
            # Calculate fees
            fee_paid = input_sum - total_sent
            fee_per_vbyte = fee_paid * 1e8 / tx["vsize"]  # Convert to sat/vB
            
            # Store transaction
            store_data(
                self.db_path,
                """INSERT OR IGNORE INTO whale_transactions 
                (txid, size, vsize, weight, fee_paid, fee_per_vbyte, total_sent, btcusd)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (txid, tx["size"], tx["vsize"], tx["weight"], fee_paid, fee_per_vbyte, total_sent, btc_price)
            )
            
            # Check for alert
            if total_sent * btc_price >= whale_alert.get_alert_threshold():
                whale_alert.detect_unusual_activity({
                    "txid": txid,
                    "sum_btc_sent": total_sent,
                    "sum_usd_sent": total_sent * btc_price,
                    "tx_in_addr": input_addresses,
                    "tx_out_addr": output_addresses
                })
        
        except Exception as e:
            print(f"Error processing tx {txid}: {str(e)}")

    def analyze_whale_behavior(self, address):
        """AI-powered behavior analysis"""
        # Fetch historical transactions
        query = """
            SELECT timestamp, total_sent, fee_per_vbyte 
            FROM whale_transactions wt
            JOIN transaction_inputs ti ON wt.txid = ti.txid
            WHERE ti.address = ?
            ORDER BY timestamp
        """
        data = fetch_data(self.db_path, query, (address,))
        
        if not data:
            return "No data available"
        
        # Prepare features for ML
        timestamps = [row[0] for row in data]
        amounts = [row[1] for row in data]
        fees = [row[2] for row in data]
        
        # Time-based features
        time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1]
        avg_time_diff = np.mean(time_diffs) if time_diffs else 0
        
        # Amount features
        avg_amount = np.mean(amounts)
        amount_std = np.std(amounts)
        
        # Fee features
        avg_fee = np.mean(fees)
        
        # Anomaly detection
        features = np.array([amounts, fees]).T
        if len(features) > 1:
            clf = IsolationForest(contamination=0.1)
            anomalies = clf.fit_predict(features)
            anomaly_count = sum(anomalies == -1)
        else:
            anomaly_count = 0
        
        # Behavior classification
        behavior = "Normal"
        if anomaly_count > len(features) * 0.3:
            behavior = "Erratic"
        elif avg_time_diff < 3600:  # 1 hour
            behavior = "Frequent Trader"
        elif avg_amount > 100:
            behavior = "Large Transactor"
        
        # Store behavior pattern
        store_data(
            self.db_path,
            """INSERT OR REPLACE INTO whale_behavior 
            (address, behavior_pattern, last_updated) 
            VALUES (?, ?, CURRENT_TIMESTAMP)""",
            (address, behavior)
        )
        
        return behavior
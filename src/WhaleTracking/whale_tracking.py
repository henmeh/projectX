import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime
from Helper.helperfunctions import fetch_btc_price
import psycopg2
from psycopg2.extras import execute_values

class WhaleTracking:
    def __init__(self, node, db_path: str, days=7):
        #if not isinstance(node, Node):
        #    raise ValueError("node must be an instance of Node or its subclass")
        self.db_params = {
            "dbname": "bitcoin_blockchain",
            "user": "postgres",
            "password": "projectX",
            "host": "localhost",
            "port": 5432,
        }

        self.node = node
        self.db_path = db_path
        self.days = days


    def connect_db(self):
        """Establish connection with optimized settings"""
        conn = psycopg2.connect(
            **self.db_params,
            application_name="BlockchainAnalytics",
            connect_timeout=10
        )
        
        # Set critical performance parameters
        with conn.cursor() as cur:
            try:
                # Stack depth solution for recursion errors
                cur.execute("SET max_stack_depth = '7680kB';")
                
                # Query optimization flags
                cur.execute("SET enable_partition_pruning = on;")
                cur.execute("SET constraint_exclusion = 'partition';")
                cur.execute("SET work_mem = '64MB';")
                
                # Transaction configuration
                cur.execute("SET idle_in_transaction_session_timeout = '5min';")
                conn.commit()
            except psycopg2.Error as e:
                print(f"Warning: Could not set session parameters: {e}")
                conn.rollback()
        return conn
    

    def process_transaction(self, txid: str, threshold: float, btc_price: float):
        """
        Process a single whale transaction and store it in the database
        Returns True if processed successfully, False otherwise
        """
        try:
            # Fetch transaction data
            tx = self.node.rpc_call("getrawtransaction", [txid, True])
            if "error" in tx or "result" not in tx:
                return False

            tx_data = tx["result"]
            total_sent = sum(out["value"] for out in tx_data.get("vout", []))

            # Skip if below threshold
            if total_sent < threshold:
                return False

            # Process inputs
            input_sum = 0
            input_records = [] # Collect records for batch insert
            for vin in tx_data.get("vin", []):
                if "txid" in vin and "vout" in vin:
                    prev_tx = self.node.rpc_call("getrawtransaction", [vin["txid"], True])
                    if "error" not in prev_tx and "result" in prev_tx:
                        prev_tx_data = prev_tx["result"]
                        prev_out = prev_tx_data["vout"][vin["vout"]]
                        input_sum += prev_out["value"]

                        # Extract address from scriptPubKey
                        if "address" in prev_out["scriptPubKey"]:
                            addr = prev_out["scriptPubKey"]["address"]
                            input_records.append((txid, addr, prev_out["value"]))

            # Process outputs
            output_records = [] # Collect records for batch insert
            for vout in tx_data.get("vout", []):
                script_pubkey = vout.get("scriptPubKey", {})
                if "address" in script_pubkey:
                    addr = script_pubkey["address"]
                    output_records.append((txid, addr, vout["value"]))

            # Calculate fees
            fee_paid = input_sum - total_sent
            fee_per_vbyte = (fee_paid * 1e8) / tx_data["vsize"] if tx_data["vsize"] > 0 else 0

            # Establish DB connection and perform inserts
            with self.connect_db() as conn:
                with conn.cursor() as cur:
                    # Insert transaction inputs
                    if input_records:
                        insert_input_query = "INSERT INTO transaction_inputs (txid, address, value) VALUES (%s, %s, %s)"
                        execute_values(cur, insert_input_query, input_records)

                    # Insert transaction outputs
                    if output_records:
                        insert_output_query = "INSERT INTO transaction_outputs (txid, address, value) VALUES (%s, %s, %s)"
                        execute_values(cur, insert_output_query, output_records)

                    # Insert or update whale transaction
                    insert_whale_tx_query = """
                        INSERT INTO whale_transactions
                        (txid, size, vsize, weight, fee_paid, fee_per_vbyte, total_sent, btcusd, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (txid) DO UPDATE SET
                            size = EXCLUDED.size,
                            vsize = EXCLUDED.vsize,
                            weight = EXCLUDED.weight,
                            fee_paid = EXCLUDED.fee_paid,
                            fee_per_vbyte = EXCLUDED.fee_per_vbyte,
                            total_sent = EXCLUDED.total_sent,
                            btcusd = EXCLUDED.btcusd,
                            timestamp = EXCLUDED.timestamp;
                    """
                    cur.execute(insert_whale_tx_query,
                                (txid, tx_data["size"], tx_data["vsize"], tx_data["weight"],
                                 fee_paid, fee_per_vbyte, total_sent, btc_price))
                conn.commit()

            return True

        except Exception as e:
            print(f"Error processing transaction {txid}: {str(e)}")
            return False
        

    def process_mempool_transactions(self, threshold: float = 100, batch_size: int = 25):
        """
        Scan mempool for whale transactions and process them in batches
        Returns count of processed transactions
        """
        # Get mempool transaction IDs
        mempool_txids = self.get_mempool_txids()
        if not mempool_txids:
            return 0
            
        btc_price = fetch_btc_price()
        processed_count = 0
        
        # Process in batches
        for i in range(0, len(mempool_txids), batch_size):
            batch = mempool_txids[i:i+batch_size]
            for txid in batch:
                if self.process_transaction(txid, threshold, btc_price):
                    processed_count += 1
        
        return processed_count


    def get_mempool_txids(self) -> list:
        """Get transaction IDs from mempool"""
        try:
            response = self.node.rpc_call("getrawmempool", [])
            return response.get("result", []) if "result" in response else []
        except Exception:
            return []


    def analyze_whale_behavior(self, address: str) -> str:
        """
        Analyze whale behavior patterns using Isolation Forest for anomaly detection
        Returns behavior classification as string
        """
        try:
            # Fetch transaction history for this address using parameterized query
            query = """
                SELECT wt.timestamp, wt.total_sent, wt.fee_per_vbyte
                FROM whale_transactions wt
                JOIN transaction_inputs ti ON wt.txid = ti.txid
                WHERE ti.address = %s
                ORDER BY wt.timestamp
            """
            data = []
            with self.connect_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (address,))
                    data = cur.fetchall()

            if not data or len(data) < 3:  # Need at least 3 transactions for analysis
                return "Insufficient Data"

            # Prepare data arrays
            timestamps = []
            amounts = []
            fees = []

            for row in data:
                # Convert timestamp string to datetime object (PostgreSQL will return datetime objects directly)
                timestamps.append(row[0])
                amounts.append(row[1])
                fees.append(row[2])

            # Calculate time-based features
            time_diffs = []
            if len(timestamps) > 1:
                for i in range(1, len(timestamps)):
                    diff = (timestamps[i] - timestamps[i-1]).total_seconds()
                    time_diffs.append(diff)
                avg_time_diff = np.mean(time_diffs) if time_diffs else 0
            else:
                avg_time_diff = 0

            # Calculate amount-based features
            avg_amount = np.mean(amounts) if amounts else 0

            # Calculate fee-based features
            avg_fee = np.mean(fees) if fees else 0

            # Prepare features for anomaly detection
            if len(amounts) > 1 and len(fees) > 1:
                features = np.array(list(zip(amounts, fees)))

                # Only run anomaly detection if we have enough data
                if len(features) > 10:
                    clf = IsolationForest(contamination=0.1, random_state=42)
                    anomalies = clf.fit_predict(features)
                    anomaly_ratio = np.sum(anomalies == -1) / len(features)
                else:
                    anomaly_ratio = 0
            else:
                anomaly_ratio = 0

            # Classify behavior based on features
            behavior = "Normal"
            if anomaly_ratio > 0.3:
                behavior = "Erratic"
            elif avg_time_diff < 3600 and len(timestamps) > 5:  # More than 1 transaction per hour
                behavior = "Frequent Trader"
            elif avg_amount >= 100:
                behavior = "Large Transactor"

            # Store behavior classification
            with self.connect_db() as conn:
                with conn.cursor() as cur:
                    insert_behavior_query = """
                        INSERT INTO whale_behavior
                        (address, behavior_pattern, last_updated)
                        VALUES (%s, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (address) DO UPDATE SET
                            behavior_pattern = EXCLUDED.behavior_pattern,
                            last_updated = EXCLUDED.last_updated;
                    """
                    cur.execute(insert_behavior_query, (address, behavior))
                conn.commit()

            return behavior

        except Exception as e:
            print(f"Error analyzing whale behavior for {address}: {str(e)}")
            return "Analysis Failed"


    def get_whale_addresses(self, min_balance: float = 500) -> list:
        """
        Identify whale addresses based on transaction history and current balance
        Returns list of whale addresses
        """
        try:
            # Get addresses with large transaction history
            query = """
                SELECT DISTINCT address
                FROM (
                    SELECT address FROM transaction_inputs
                    UNION ALL
                    SELECT address FROM transaction_outputs
                ) AS combined_addresses
                WHERE address IN (
                    SELECT ti.address FROM transaction_inputs ti
                    JOIN whale_transactions wt ON ti.txid = wt.txid
                    WHERE wt.total_sent > 10
                    GROUP BY ti.address
                    HAVING COUNT(ti.address) > 3
                )
            """
            candidate_addresses = []
            with self.connect_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    candidate_addresses = [row[0] for row in cur.fetchall()]

            # Filter by current balance (Assuming get_address_balance is implemented elsewhere
            # and can fetch from the node or a more comprehensive database)
            whale_addresses = []
            for address in candidate_addresses:
                # This part needs an actual implementation to fetch the balance.
                # For now, it remains commented out or uses a placeholder.
                # balance = self.get_address_balance(address)
                balance = 0 # Placeholder: Replace with actual balance fetching
                if balance >= min_balance:
                    whale_addresses.append(address)

            return whale_addresses

        except Exception as e:
            print(f"Error getting whale addresses: {e}")
            return []

    #def track_whale_balances(self, addresses: list):
    #    """Track and store balance history for whale addresses"""
    #    for address in addresses:
    #        balance = self.get_address_balance(address)
    #        if balance > 0:  # Only store if we have a positive balance
    #            store_data(
    #                self.db_path,
    #                """INSERT OR REPLACE INTO whale_balance_history 
    #                (address, timestamp, confirmed_balance) 
    #                VALUES (?, CURRENT_TIMESTAMP, ?)""",
    #                (address, balance)
    #            )
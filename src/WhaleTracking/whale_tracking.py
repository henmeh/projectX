import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime
from Helper.helperfunctions import create_table, fetch_btc_price, store_data, fetch_data_params, fetch_data
import psycopg2
from psycopg2.extras import execute_values

class WhaleTracking:
    def __init__(self, node, days=7):
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
        
    
    def process_transaction(self, txid: list, threshold: float, btc_price: float):
        """
        Process a single whale transaction and store it in the database
        Returns True if processed successfully, False otherwise
        """
        conn = self.connect_db()
        cursor = conn.cursor()

        try:            
            # Fetch transaction data
            transactions = self.node.rpc_batch_call("getrawtransaction", txid)
            
            for transaction in transactions:
                total_sent = sum(float(transaction_output["value"]) for transaction_output in transaction.get("vout", []))

                # Skip if below threshold
                if total_sent < threshold:
                    return False
                
                # Process inputs
                input_sum = 0
                input_addresses = []
                current_txid = transaction.get("txid", 0)
                vin_txids = []
                vin_vouts = []

                for transaction_input in transaction.get("vin", []):
                    if "txid" in transaction_input and "vout" in transaction_input:
                        vin_txids.append(transaction_input["txid"])
                        vin_vouts.append(transaction_input["vout"])

                prev_txs = self.node.rpc_batch_call("getrawtransaction", vin_txids)
                    
                i = 0
                for prev_tx in prev_txs:
                    prev_out = prev_tx["vout"][vin_vouts[i]]
                    input_sum += float(prev_out["value"])
                    i += 1
                    
                    if "address" in prev_out["scriptPubKey"]:
                        addr = prev_out["scriptPubKey"]["address"]
                    else:
                        addr = prev_out["scriptPubKey"]["asm"]
                    
                    input_addresses.append(addr)
                    insert_data = [(current_txid, addr, float(prev_out["value"]))]
                    
                    execute_values(
                        cursor,
                        "INSERT INTO transactions_inputs (txid, address, value) VALUES %s",
                        insert_data
                    )

                output_addresses = []
                for vout in transaction.get("vout", []):
                    script_pubkey = vout.get("scriptPubKey", {})
                    if "address" in script_pubkey:
                        addr = script_pubkey["address"]
                    else:
                        addr = script_pubkey["asm"]
                    output_addresses.append(addr)
                    insert_data = [(current_txid, addr, float(vout["value"]))]
                    
                    execute_values(
                        cursor,
                        "INSERT INTO transactions_outputs (txid, address, value) VALUES %s",
                        insert_data
                    )
            
                fee_paid = input_sum - total_sent
                fee_per_vbyte = (fee_paid * 1e8) / transaction["vsize"] if transaction["vsize"] > 0 else 0
                insert_data = [(current_txid, datetime.now(), transaction["size"], transaction["vsize"], transaction["weight"], fee_paid, fee_per_vbyte, total_sent, btc_price)] 
                
                execute_values(
                    cursor,
                    """INSERT INTO whale_transactions 
                    (txid, timestamp, size, vsize, weight, fee_paid, fee_per_vbyte, total_sent, btcusd)
                    VALUES %s
                    ON CONFLICT (txid) DO UPDATE SET
                    timestamp = EXCLUDED.timestamp,
                    size = EXCLUDED.size,
                    vsize = EXCLUDED.vsize,
                    weight = EXCLUDED.weight,
                    fee_paid = EXCLUDED.fee_paid,
                    fee_per_vbyte = EXCLUDED.fee_per_vbyte,
                    total_sent = EXCLUDED.total_sent,
                    btcusd = EXCLUDED.btcusd""",
                    insert_data
                )
                conn.commit()

            return True
        except Exception as e:
            print(f"Error processing transaction {txid}: {str(e)}")
            return False
    
    
    def process_mempool_transactions(self, threshold: float = 100, batch_size: int = 100) -> int:
        """
        Scan mempool for whale transactions and process them in batches
        Returns count of processed transactions
        """
        # Get mempool transaction IDs
        mempool_txids = self.get_mempool_txids()
        if not mempool_txids:
            return 0
            
        btc_price = 0#fetch_btc_price()
        processed_count = 0
        
        for i in range(0, len(mempool_txids), batch_size):
        #for i in range(0, 2, batch_size):
            txid_batch = mempool_txids[i:i+batch_size]
            if self.process_transaction(txid_batch, threshold, btc_price):
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
                WHERE ti.address = ?
                ORDER BY wt.timestamp
            """
            data = fetch_data_params(self.db_path, query, (address,))
            
            if not data or len(data) < 3:  # Need at least 3 transactions for analysis
                return "Insufficient Data"
            
            # Prepare data arrays
            timestamps = []
            amounts = []
            fees = []
            
            for row in data:
                # Convert timestamp string to datetime object
                timestamp_str = row[0]
                if '.' in timestamp_str:
                    dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
                else:
                    dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                timestamps.append(dt)
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
            elif avg_amount >= 100:  # Changed to >= for better threshold handling
                behavior = "Large Transactor"
            
            # Store behavior classification
            store_data(
                self.db_path,
                """INSERT OR REPLACE INTO whale_behavior 
                (address, behavior_pattern, last_updated) 
                VALUES (?, ?, CURRENT_TIMESTAMP)""",
                (address, behavior)
            )
            
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
                )
                WHERE address IN (
                    SELECT address FROM transaction_inputs
                    JOIN whale_transactions ON transaction_inputs.txid = whale_transactions.txid
                    WHERE whale_transactions.total_sent > 10
                    GROUP BY address
                    HAVING COUNT(*) > 3
                )
            """
            candidate_addresses = [row[0] for row in fetch_data(self.db_path, query)]
            
            # Filter by current balance
            whale_addresses = []
            for address in candidate_addresses:
                balance = self.get_address_balance(address)
                if balance >= min_balance:
                    whale_addresses.append(address)
            
            return whale_addresses
        
        except Exception:
            return []


    #def get_address_balance(self, address: str) -> float:
    #    """Get current balance of an address"""
    #    try:
    #        response = self.node.rpc_call("getaddressbalance", [{"addresses": [address]}])
    #        if "result" in response and "balance" in response["result"]:
    #            # Balance is returned in satoshis, convert to BTC
    #            return response["result"]["balance"] / 1e8
    #        return 0
    #    except Exception:
    #        return 0

    def get_address_balance(self, address):
        """
        Retrieve the balance of a Bitcoin address from the database
        
        Args:
            db_params (dict): Database connection parameters
            address (str): Bitcoin address to query
        
        Returns:
            int: Address balance in satoshis (or 0 if no UTXOs found)
        """
        try:
            # Connect to the database
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()
            
            # SQL query to sum unspent UTXOs
            query = sql.SQL("""
                SELECT COALESCE(SUM(value), 0)
                FROM utxos
                WHERE address = %s AND spent = false
            """)
            
            # Execute the query
            cursor.execute(query, (address,))
            balance = cursor.fetchone()[0]
            
            return balance
            
        except psycopg2.Error as e:
            print(f"Database error: {e}")
            return None
        finally:
            # Ensure connection is closed even if error occurs
            if conn:
                conn.close()
                

    def track_whale_balances(self, addresses: list):
        """Track and store balance history for whale addresses"""
        for address in addresses:
            balance = self.get_address_balance(address)
            if balance > 0:  # Only store if we have a positive balance
                store_data(
                    self.db_path,
                    """INSERT OR REPLACE INTO whale_balance_history 
                    (address, timestamp, confirmed_balance) 
                    VALUES (?, CURRENT_TIMESTAMP, ?)""",
                    (address, balance)
                )
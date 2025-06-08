import json
import time
import psycopg2
from psycopg2.extras import execute_values
import datetime
import time
from Helper.helperfunctions import fetch_historical_btc_price

class BlockchainStoring:
    def __init__(self, node):
        self.node = node
        self.db_params = {
            "dbname": "bitcoin_blockchain",
            "user": "postgres",
            "password": "projectX",
            "host": "localhost",
            "port": 5432,
        }
        self.btc_price_cache = {}
        self.total_processed = 0
        self.start_time = time.time()

    #def connect_db(self):
    #    """Establish connection with optimized settings"""
    #    conn = psycopg2.connect(**self.db_params)
    #    conn.autocommit = False
    #    return conn

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
                cur.execute("SET max_stack_depth = '4096kB';")
                
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


    def get_btc_price(self, timestamp):
        """Cached BTC price fetcher"""
        date = time.strftime("%Y-%m-%d", time.gmtime(timestamp))
        if date not in self.btc_price_cache:
            self.btc_price_cache[date] = fetch_historical_btc_price(timestamp)
        return self.btc_price_cache[date]


    def process_block(self, block_height: int):
        """Process a block and store its data"""
        try:
            # Fetch block data
            block_hash = self.node.rpc_call("getblockhash", [block_height])["result"]
            block_data = self.node.rpc_call("getblock", [block_hash, 2])["result"]
            
            transactions, utxos, spent_utxos, address_changes = self.extract_block_data(block_data, block_height)

            block_time = block_data["time"]
            self.store_data(transactions, utxos, spent_utxos, address_changes, block_time)
            
            return len(transactions)
        
        except Exception as e:
            print(f"❌ Error processing block {block_height}: {e}")
            # Delete partially processed block
            self.delete_existing_block_data(block_height)
            return False


    def extract_block_data(self, block_data, block_height):
        """Extract relevant data from block"""
        tx_list = block_data["tx"]
        block_time = block_data["time"]
        btc_price_usd = self.get_btc_price(block_time)

        transactions = []
        utxos = []
        spent_utxos = []
        address_changes = {}

        for tx in tx_list:
            txid = tx["txid"]
            transactions.append((
                txid, block_height, block_time, 
                tx.get("fee", 0), tx["size"], tx["weight"], 
                json.dumps(tx), btc_price_usd, datetime.datetime.fromtimestamp(block_time)
            ))

            for vin in tx["vin"]:
                if "txid" in vin and "vout" in vin:
                    spent_utxos.append((vin["txid"], vin["vout"], txid))

            for vout in tx["vout"]:
                script_pubkey = vout.get("scriptPubKey", {})
                address = self.get_output_address(script_pubkey)
                output_type = script_pubkey["type"]
                value = int(vout["value"] * 100000000)
                
                utxos.append((
                    txid, vout["n"], address, value,
                    block_height, False, block_time,
                    btc_price_usd, output_type, datetime.datetime.fromtimestamp(block_time)
                ))
                
                address_changes[address] = address_changes.get(address, 0) + value

        return transactions, utxos, spent_utxos, address_changes


    def get_output_address(self, script_pubkey):
        """Unified address extraction"""
        if 'address' in script_pubkey:
            return script_pubkey['address']
        if 'addresses' in script_pubkey and script_pubkey['addresses']:
            return script_pubkey['addresses'][0]
        else:
            return script_pubkey['asm']


    def store_data(self, transactions, utxos, spent_utxos, address_changes, block_time):
        """Optimized storage with batch processing"""
        conn = self.connect_db()
        cursor = conn.cursor()
        
        try:
            execute_values(
                cursor,
                """INSERT INTO transactions (txid, block_height, timestamp, fee, 
                    size, weight, raw_tx, btc_price_usd, date) 
                VALUES %s """,
                transactions
            )
            
            execute_values(
                    cursor,
                    """INSERT INTO utxos (txid, vout, address, value, block_height, 
                        spent, timestamp, btc_price_usd, output_type, date) 
                    VALUES %s """,
                    utxos
                )
            
            # Process spent UTXOs in batches
            query = """
                    UPDATE utxos 
                    SET spent = TRUE, spent_in_txid = data.spent_in_txid
                    FROM (VALUES %s) AS data(txid, vout, spent_in_txid)
                    WHERE utxos.txid = data.txid AND utxos.vout = data.vout
                """
            execute_values(cursor, query, spent_utxos, template="(%s, %s, %s)")

            # Update address balances in batches
            address_items = list(address_changes.items())
            data = [(addr, change, block_time, datetime.datetime.fromtimestamp(block_time)) for addr, change in address_items]
            execute_values(
                    cursor,
                    """INSERT INTO addresses (address, balance, last_seen, date) 
                    VALUES %s
                    ON CONFLICT (address) DO UPDATE SET 
                        balance = addresses.balance + EXCLUDED.balance,
                        last_seen = EXCLUDED.last_seen,
                        date = EXCLUDED.date""",
                    data,
                )
            conn.commit()
                
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()


    def sync_blocks(self, start_height: int, end_height: int = None):
        """Sync blocks with optimized performance"""
        if end_height is None:
            end_height = self.node.rpc_call("getblockcount", [])["result"]
        
        print(f"🚀 Syncing blocks {start_height}-{end_height}")
        self.total_processed = 0
        
        for height in range(start_height, end_height + 1):
            success = self.process_block(height)
            print(f"🚀 Syncing block {height}/{success} Transactions from {end_height}")

            if not success:
                print(f"⏸️ Stopping sync at block {height}")
                return height  # Return last successful block

        total_duration = time.time() - self.start_time
        total_rate = self.total_processed / total_duration
        print(f"✅ Blockchain sync complete! "
              f"Processed {self.total_processed} transactions in {total_duration/60:.1f} minutes "
              f"({total_rate:.1f} tx/s)")
        return end_height


    def get_address_balance(self, address: str):
        """Get current balance of an address"""
        with self.connect_db() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT balance FROM addresses WHERE address = %s",
                    (address,)
                )
                result = cursor.fetchone()
                return result[0] if result else 0


    def get_latest_stored_block(self):
        """Get highest block stored in database"""
        with self.connect_db() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT MAX(block_height) FROM transactions
                """)
                return cursor.fetchone()[0]


    def delete_existing_block_data(self, block_height):
        """Efficient block deletion for partitioned tables"""
        with self.connect_db() as conn:
            with conn.cursor() as cursor:
                print(f"🧹 Deleting data for block {block_height}")
                
                try:
                    # Delete transactions (partition pruning will make this fast)
                    cursor.execute("""
                        DELETE FROM transactions 
                        WHERE block_height = %s
                    """, (block_height,))
                    
                    # Delete UTXOs
                    cursor.execute("""
                        DELETE FROM utxos 
                        WHERE block_height = %s
                    """, (block_height,))
                    
                    # Reset addresses balances
                    cursor.execute("""
                        WITH address_changes AS (
                            SELECT address, SUM(-value) as delta
                            FROM utxos
                            WHERE block_height = %s AND address IS NOT NULL
                            GROUP BY address
                        )
                        UPDATE addresses a
                        SET balance = a.balance + ac.delta
                        FROM address_changes ac
                        WHERE a.address = ac.address
                    """, (block_height,))
                    
                    conn.commit()
                    print(f"🧹 Deleted data for block {block_height}")
                except Exception as e:
                    conn.rollback()
                    print(f"❌ Error deleting block {block_height}: {e}")
    
  
    def get_block_data(self, block_height):
        """Get basic block information"""
        with self.connect_db() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT 
                        MIN(timestamp) AS block_time,
                        COUNT(*) AS tx_count,
                        SUM(fee) AS total_fees,
                        AVG(btc_price_usd) AS avg_btc_price
                    FROM transactions
                    WHERE block_height = %s
                """, (block_height,))
                return cursor.fetchone()


    def get_address_history(self, address):
        """Get transaction history for an address"""
        with self.connect_db() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    (SELECT 
                        t.txid,
                        t.block_height,
                        t.timestamp,
                        'in' AS direction,
                        u.value,
                        t.fee,
                        t.btc_price_usd
                    FROM utxos u
                    JOIN transactions t ON u.txid = t.txid
                    WHERE u.address = %s)
                    
                    UNION ALL
                    
                    (SELECT 
                        t.txid,
                        t.block_height,
                        t.timestamp,
                        'out' AS direction,
                        -u.value AS value,
                        t.fee,
                        t.btc_price_usd
                    FROM utxos u
                    JOIN transactions t ON u.spent_in_txid = t.txid
                    WHERE u.address = %s)
                    
                    ORDER BY timestamp DESC
                """, (address, address))
                return cursor.fetchall()
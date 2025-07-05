import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
import json
import time
from datetime import datetime
import psycopg2

class Mempool:
    def __init__(self, node):
        #if not isinstance(node, Node):
        #    raise ValueError("node must be an instance of Node or its subclass")
            
        self.node = node
        self.db_params = {
            "dbname": "bitcoin_blockchain",
            "user": "postgres",
            "password": "projectX",
            "host": "localhost",
            "port": 5432,
        }
        self.last_fetch_time = 0
        self.fee_cache = None
        self.cache_duration = 60


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

        
    def get_mempool_feerates(self, block_vsize_limit: int = 1_000_000) -> dict:
        current_time = time.time()
        
        # Return cached data if valid
        if self.fee_cache and (current_time - self.last_fetch_time) < self.cache_duration:
            return self.fee_cache
        
        conn = None
        try:
            conn = self.connect_db()
            cursor = conn.cursor()

            # Get fee histogram from node
            fee_histogram = self.node.electrum_request("mempool.get_fee_histogram")["result"]
            
            if not fee_histogram:
                raise ValueError("Empty fee histogram response")
            
            total_vsize = 0
            percentiles = {
                25: {"vsize": block_vsize_limit * 0.25, "fee": None},
                50: {"vsize": block_vsize_limit * 0.50, "fee": None},
                75: {"vsize": block_vsize_limit * 0.75, "fee": None}
            }
            
            # Calculate fee percentiles
            for fee_rate, vsize in fee_histogram:
                total_vsize += vsize
                for pct in percentiles:
                    if percentiles[pct]["fee"] is None and total_vsize >= percentiles[pct]["vsize"]:
                        percentiles[pct]["fee"] = fee_rate
                if total_vsize >= block_vsize_limit:
                    break
            
            # Prepare result
            result = {
                "fast": percentiles[25]["fee"] or 1,
                "medium": percentiles[50]["fee"] or 1,
                "low": percentiles[75]["fee"] or 1,
                "histogram": fee_histogram
            }
            
            # Update cache
            self.fee_cache = result
            self.last_fetch_time = current_time
            
            # Insert into database
            cursor.execute(
                "INSERT INTO mempool_fee_histogram (timestamp, histogram, fast_fee, medium_fee, low_fee) "
                "VALUES (%s, %s, %s, %s, %s)",
                (datetime.now(), json.dumps(fee_histogram), result["fast"], result["medium"], result["low"])
            )
            conn.commit()
            
            return result
            
        except Exception as e:
            print(e)
            # FALLBACK: Use last known data from database
            if conn:
                cursor = conn.cursor()
                try:
                    cursor.execute(
                        "SELECT histogram, fast_fee, medium_fee, low_fee "
                        "FROM mempool_fee_histogram "
                        "ORDER BY timestamp DESC LIMIT 1"
                    )
                    last_data = cursor.fetchone()
                    
                    if last_data:
                        return {
                            "fast": last_data[1],
                            "medium": last_data[2],
                            "low": last_data[3],
                            "histogram": json.loads(last_data[0])
                        }
                except Exception as inner_e:
                    print(f"Fallback query failed: {inner_e}")
            
            # Ultimate fallback if everything fails
            return {
                "fast": 10,
                "medium": 5,
                "low": 1,
                "histogram": []
            }
            
        finally:
            # Ensure connection is always closed
            if conn:
                conn.close()
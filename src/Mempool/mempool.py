import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
import json
import time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from Helper.helperfunctions import store_data, fetch_data
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_values

class Mempool:
    def __init__(self, node, db_path: str):
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
        self.db_path = db_path
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
                "fast": percentiles[25]["fee"] or 0,
                "medium": percentiles[50]["fee"] or 0,
                "low": percentiles[75]["fee"] or 0,
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
                
    
    def predict_fee_rates(self, time_horizon: int = 10) -> dict:
        """Predict future fee rates using polynomial regression"""
        conn = self.connect_db()
        cursor = conn.cursor()
        try:
            # Fetch historical data
            query = """
                        SELECT timestamp, fast_fee, medium_fee, low_fee FROM mempool_fee_histogram
                        WHERE timestamp >= datetime('now', '-2 hours') ORDER BY timestamp ASC   
                    """
            execute_values(cursor, query, [])
            data = cursor.fetchall()
            
            if not data or len(data) < 5:
                current = self.get_mempool_feerates()
                return {
                    "fast": current["fast"],
                    "medium": current["medium"],
                    "low": current["low"]
                }
            
            # Prepare data arrays
            timestamps = []
            fast_fees = []
            medium_fees = []
            low_fees = []
            
            # Convert string timestamps to datetime objects
            for row in data:
                # SQLite returns timestamps as strings
                timestamp_str = row[0]
                # Handle different timestamp formats
                if '.' in timestamp_str:
                    dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
                else:
                    dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                timestamps.append(dt)
                fast_fees.append(row[1])
                medium_fees.append(row[2])
                low_fees.append(row[3])
            
            # Convert to minutes since first timestamp
            min_timestamp = min(timestamps)
            minutes = [(ts - min_timestamp).total_seconds() / 60 for ts in timestamps]
            
            # Create polynomial features
            poly = PolynomialFeatures(degree=2)
            X = np.array(minutes).reshape(-1, 1)
            X_poly = poly.fit_transform(X)
            
            # Train models and predict
            predictions = {}
            for fee_type, values in [("fast", fast_fees), ("medium", medium_fees), ("low", low_fees)]:
                model = LinearRegression()
                model.fit(X_poly, values)
                
                future_minute = minutes[-1] + time_horizon
                future_poly = poly.transform([[future_minute]])
                pred_value = model.predict(future_poly)[0]
                predictions[fee_type] = max(1, round(pred_value, 2))
            
            # Store prediction
            execute_values(
                cursor,
                "INSERT INTO fee_predictions (prediction_time, fast_fee_pred, medium_fee_pred, low_fee_pred) VALUES (?, ?, ?, ?)",
                (time_horizon, predictions["fast"], predictions["medium"], predictions["low"])
            )
            conn.commit()
            
            return predictions
        
        except Exception as e:
            print(f"Error predicting fee rates: {str(e)}")
            current = self.get_mempool_feerates()
            return {"fast": current["fast"], "medium": current["medium"], "low": current["low"]}
import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
import json
import time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from Helper.helperfunctions import create_table, store_data, fetch_data
from datetime import datetime

class Mempool:
    def __init__(self, node, db_path: str):
        #if not isinstance(node, Node):
        #    raise ValueError("node must be an instance of Node or its subclass")
            
        self.node = node
        self.db_path = db_path
        self.last_fetch_time = 0
        self.fee_cache = None
        self.cache_duration = 60
        
        # Initialize database tables
        create_table(db_path, '''CREATE TABLE IF NOT EXISTS mempool_fee_histogram (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            histogram TEXT,
            fast_fee REAL,
            medium_fee REAL,
            low_fee REAL)''')
        
        create_table(db_path, '''CREATE TABLE IF NOT EXISTS fee_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            prediction_time INTEGER,
            fast_fee_pred REAL,
            medium_fee_pred REAL,
            low_fee_pred REAL)''')

    def get_mempool_feerates(self, block_vsize_limit: int = 1_000_000) -> dict:
        current_time = time.time()
        if self.fee_cache and (current_time - self.last_fetch_time) < self.cache_duration:
            return self.fee_cache
        
        try:
            # Get fee histogram from node implementation
            fee_histogram = self.node.electrum_request("mempool.get_fee_histogram")["result"]
            
            if not fee_histogram:
                raise ValueError("Empty fee histogram response")
            
            total_vsize = 0
            percentiles = {
                25: {"vsize": block_vsize_limit * 0.25, "fee": None},
                50: {"vsize": block_vsize_limit * 0.50, "fee": None},
                75: {"vsize": block_vsize_limit * 0.75, "fee": None}
            }
            
            for fee_rate, vsize in fee_histogram:
                total_vsize += vsize
                for pct in percentiles:
                    if not percentiles[pct]["fee"] and total_vsize >= percentiles[pct]["vsize"]:
                        percentiles[pct]["fee"] = fee_rate
                if total_vsize >= block_vsize_limit:
                    break
            
            result = {
                "fast": percentiles[25]["fee"],
                "medium": percentiles[50]["fee"],
                "low": percentiles[75]["fee"],
                "histogram": fee_histogram
            }
            
            self.fee_cache = result
            self.last_fetch_time = current_time
            
            store_data(
                self.db_path,
                "INSERT INTO mempool_fee_histogram (histogram, fast_fee, medium_fee, low_fee) VALUES (?, ?, ?, ?)",
                (json.dumps(fee_histogram), result["fast"], result["medium"], result["low"])
            )
            
            return result
        
        except Exception:
            # Fallback to last known data
            last_data = fetch_data(
                self.db_path,
                "SELECT histogram, fast_fee, medium_fee, low_fee FROM mempool_fee_histogram ORDER BY timestamp DESC LIMIT 1"
            )
            if last_data:
                return {
                    "fast": last_data[0][2],
                    "medium": last_data[0][3],
                    "low": last_data[0][4],
                    "histogram": json.loads(last_data[0][1])
                }
            return {"fast": 10, "medium": 5, "low": 1, "histogram": []}

    
    def predict_fee_rates(self, time_horizon: int = 10) -> dict:
        """Predict future fee rates using polynomial regression"""
        try:
            # Fetch historical data
            data = fetch_data(
                self.db_path,
                "SELECT timestamp, fast_fee, medium_fee, low_fee FROM mempool_fee_histogram "
                "WHERE timestamp >= datetime('now', '-2 hours') ORDER BY timestamp ASC"
            )
            
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
            store_data(
                self.db_path,
                "INSERT INTO fee_predictions (prediction_time, fast_fee_pred, medium_fee_pred, low_fee_pred) VALUES (?, ?, ?, ?)",
                (time_horizon, predictions["fast"], predictions["medium"], predictions["low"])
            )
            
            return predictions
        
        except Exception as e:
            print(f"Error predicting fee rates: {str(e)}")
            current = self.get_mempool_feerates()
            return {"fast": current["fast"], "medium": current["medium"], "low": current["low"]}
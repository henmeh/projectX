import pandas as pd
from sqlalchemy import create_engine, text, Table, MetaData, Column, DateTime, Numeric, String, BigInteger, Integer, insert, inspect, UniqueConstraint
from datetime import datetime, timedelta, timezone
import json
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MempoolAnalyzer:
    """
    A class to fetch mempool transaction data, analyze it by transaction value,
    and store the aggregated insights into a database.
    """

    def __init__(self, db_connection_string,
                 whale_transactions_table_name='whale_transactions',
                 mempool_insights_table_name='mempool_value_insights'): # Analyze transactions from the last X hours
        self.db_connection_string = db_connection_string
        self.whale_transactions_table_name = whale_transactions_table_name
        self.mempool_insights_table_name = mempool_insights_table_name

        self.engine = create_engine(self.db_connection_string)
        self.metadata = MetaData()


    def analyze_mempool_transactions(self):
        """
        Fetches recent transactions from the whale_transactions table,
        then calculates the distribution of vsize, average fee, and transaction count
        across different total_sent amount buckets.
        """
        
        try:
            with self.engine.connect() as conn:
                # Fetch recent transactions from the whale_transactions table
                # We need: vsize, fee_per_vbyte, total_sent, and timestamp for filtering
                query = text(f"""
                    SELECT
                        vsize,
                        fee_per_vbyte,
                        total_sent
                    FROM
                        {self.whale_transactions_table_name}
                    ORDER BY
                        timestamp DESC;
                """)
                # Use parameters for the query to prevent SQL injection
                result = conn.execute(query, )
                mempool_txs = result.fetchall()

        except Exception as e:
            logging.error(f"Error fetching transactions from PostgreSQL: {e}")
            logging.exception("Detailed traceback for DB fetch error:")
            return None

        # --- Define Amount Buckets (in BTC) ---
        # Using a list of tuples for ordered processing and clear range definitions
        amount_buckets = [
            ("0-1 BTC", 0, 1),
            ("1-10 BTC", 1, 10),
            ("10-50 BTC", 10, 50),
            ("50-100 BTC", 50, 100),
            (">100 BTC", 100, float('inf')) # Use float('inf') for the upper bound
        ]

        # Initialize aggregation dictionaries
        aggregated_data = {
            bucket_name: {'total_vsize': 0, 'fees': [], 'count': 0}
            for bucket_name, _, _ in amount_buckets
        }

        # --- Process Transactions ---
        for tx in mempool_txs:
            vsize = tx[0]
            fee_per_vbyte = tx[1]
            total_sent = tx[2]

            for bucket_name, lower_bound, upper_bound in amount_buckets:
                if lower_bound <= total_sent < upper_bound:
                    aggregated_data[bucket_name]['total_vsize'] += vsize
                    aggregated_data[bucket_name]['fees'].append(fee_per_vbyte)
                    aggregated_data[bucket_name]['count'] += 1
                    break # Move to the next transaction once a bucket is found

        # --- Finalize Aggregations and Prepare for Storage ---
        insights_to_store = []
        generated_at_timestamp = datetime.now(timezone.utc) # Consistent timestamp for this analysis run

        for bucket_name, data in aggregated_data.items():
            avg_fee = np.mean(data['fees']) if data['fees'] else 0 # Handle empty lists
            
            insights_to_store.append({
                'generated_at': generated_at_timestamp,
                'amount_range': bucket_name,
                'total_vsize_bytes': data['total_vsize'],
                'avg_fee_per_vbyte': float(avg_fee),
                'transaction_count': data['count']
            })
        
        # Store the insights in the database
        self._store_mempool_insights(insights_to_store)
        
        logging.info("Mempool transaction analysis completed and insights stored.")
        # Return the insights for immediate use or verification
        return insights_to_store

    def _store_mempool_insights(self, insights_data):
        """Stores the calculated mempool insights into the database."""
        if not insights_data:
            logging.warning("No mempool insights data to store.")
            return

        # Define the mempool_value_insights table schema
        mempool_insights_table = Table(
            self.mempool_insights_table_name, self.metadata,
            Column('id', BigInteger, primary_key=True, autoincrement=True),
            Column('generated_at', DateTime(timezone=True), nullable=False),
            Column('amount_range', String(50), nullable=False),
            Column('total_vsize_bytes', BigInteger, nullable=False),
            Column('avg_fee_per_vbyte', Numeric, nullable=False),
            Column('transaction_count', Integer, nullable=False),
            UniqueConstraint('generated_at', 'amount_range', name='uq_insight_gen_range')
        )

        try:
            with self.engine.connect() as conn:
                conn.execute(insert(mempool_insights_table), insights_data)
                conn.commit()

            logging.info(f"Successfully stored {len(insights_data)} mempool insights to '{self.mempool_insights_table_name}'.")
        except Exception as e:
            logging.error(f"Error storing mempool insights: {e}")
            logging.exception("Detailed traceback for DB storage error:")
            raise # Re-raise to indicate failure

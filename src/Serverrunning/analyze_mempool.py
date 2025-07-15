import time
import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
from Mempool.mempool_analyzer import MempoolAnalyzer
import logging

# Example Usage (for testing the backend logic)
if __name__ == "__main__":
    # IMPORTANT: Replace with your actual database connection string
    DB_CONN_STR = 'postgresql://postgres:projectX@localhost:5432/bitcoin_blockchain'
    WHALE_TX_TABLE = 'whale_transactions'
    INSIGHTS_TABLE = 'mempool_value_insights'

    while True:
        try:
            analyzer = MempoolAnalyzer(
                db_connection_string=DB_CONN_STR,
                whale_transactions_table_name=WHALE_TX_TABLE,
                mempool_insights_table_name=INSIGHTS_TABLE,
            )

            # Run the analysis and store the results
            print("Running mempool analysis...")
            latest_insights = analyzer.analyze_mempool_transactions()
            
        except RuntimeError as e:
            logging.critical(f"Application terminated due to critical setup error: {e}")
        except Exception as e:
            logging.critical(f"An unhandled error occurred during execution: {e}")
            logging.exception("Unhandled exception traceback:")
        time.sleep(60)
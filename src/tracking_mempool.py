import sqlite3
import time
import requests
from Mempool.mempool import Mempool
from Chain.chain import Chain


# SQLite database connection
DB_PATH = "/media/henning/Volume/Programming/projectX/src/mempol_data.db"

def create_table():
    """Creates the SQLite table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS mempool_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        fast_fee REAL,
                        medium_fee REAL,
                        low_fee REAL,
                        mempool_size INTEGER,
                        tx_count INTEGER,
                        block_time INTEGER)''')
    conn.commit()
    conn.close()

def fetch_mempool_data():
    """Fetches mempool fee rates and transaction data."""
    try:
        mempool = Mempool()
        chain = Chain()
        fee_rates = mempool.get_mempool_feerates()
        fast_fee = fee_rates[int(len(fee_rates) * 0.25)] if len(fee_rates) > 10 else max(fee_rates)
        medium_fee = fee_rates[int(len(fee_rates) * 0.5)] if len(fee_rates) > 2 else fee_rates[0]  # Median
        low_fee = fee_rates[-1]
        
        mempool_size, tx_count = mempool.get_mempool_stats()
        
        latest_block = chain.get_block_height()
        block_time = latest_block
        
        return fast_fee, medium_fee, low_fee, mempool_size, tx_count, block_time
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def store_data():
    """Fetches mempool data and stores it in the SQLite database."""
    data = fetch_mempool_data()
    if data:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO mempool_data (fast_fee, medium_fee, low_fee, 
                          mempool_size, tx_count, block_time) 
                          VALUES (?, ?, ?, ?, ?, ?)''', data)
        conn.commit()
        conn.close()
        print(f"✅ Data stored at {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    create_table()
    while True:
        store_data()
        time.sleep(60)  # Run every 60 seconds

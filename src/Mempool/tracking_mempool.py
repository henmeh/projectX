import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from node_data import ELECTRUM_HOST
import sqlite3
import time
import requests


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
        url = f"http://{ELECTRUM_HOST}:50001"  # Change to your Electrum server
        # Fetching fee estimates (modify this to match your actual API call)
        fees = requests.get("https://mempool.space/api/v1/fees/recommended").json()
        fast_fee = fees["fastestFee"]
        medium_fee = fees["halfHourFee"]
        low_fee = fees["hourFee"]
        
        # Fetching mempool stats
        mempool = requests.get("https://mempool.space/api/mempool").json()
        mempool_size = mempool["bytes"]
        tx_count = mempool["count"]
        
        # Fetching latest block info
        latest_block = requests.get("https://mempool.space/api/blocks/tip/height").json()
        block_time = latest_block  # You may need to process this differently
        
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

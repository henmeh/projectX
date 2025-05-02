import bitcoinlib
import requests
import sqlite3
from datetime import datetime, timedelta
import json
import time
import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
from node_data import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, COINCAP_API_KEY


def create_table(path_to_db:str, sql_command:str):
    """Creates the SQLite table if it doesn't exist."""
    DB_PATH = path_to_db
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(sql_command)
    conn.commit()
    conn.close()


def store_data(path_to_db:str, sql_command:str, data:tuple):
    """Stores data in the SQLite database."""
    conn = sqlite3.connect(path_to_db)
    cursor = conn.cursor()
    cursor.execute(sql_command, data)
    conn.commit()
    conn.close()


def fetch_data(path_to_db:str, sql_command:str) -> list:
    """Fetches data from the SQLite database."""
    conn = sqlite3.connect(path_to_db)
    cursor = conn.cursor()
    cursor.execute(sql_command)
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_existing_txids(path_to_db: str, txids: list) -> set:
    """Fetch existing txids from the database to avoid duplicate processing."""
    query = f"SELECT txid FROM mempool_transactions WHERE txid IN ({','.join(['?']*len(txids))})"
    
    try:
        conn = sqlite3.connect(path_to_db)
        cursor = conn.cursor()
        cursor.execute(query, txids)
        existing_txids = {row[0] for row in cursor.fetchall()}
        conn.close()
        return existing_txids
    except Exception as e:
        print(f"Database Error: {e}")
        return set()


def fetch_whale_transactions(db_mempool_transactions_path: str, days: int) -> list:
        """
        Fetches whale transactions from the last `days` days.
        Parses JSON-encoded addresses and returns structured data.
        """
        conn = sqlite3.connect(db_mempool_transactions_path)
        cursor = conn.cursor()

        # Define the date range
        start_date = datetime.now() - timedelta(days=days)

        # SQL Query to fetch transactions within the given time frame
        query = """
        SELECT timestamp, txid, tx_in_addr, tx_out_addr, total_sent, fee_paid, fee_per_vbyte
        FROM mempool_transactions
        WHERE timestamp >= ?
        """
        cursor.execute(query, (start_date.strftime("%Y-%m-%d %H:%M:%S"),))
        
        transactions = []
        
        for row in cursor.fetchall():
            timestamp, txid, tx_in_addr, tx_out_addr, total_sent, fee_paid, fee_per_vbyte = row
            
            # Parse JSON-encoded addresses
            try:
                tx_in_addr = json.loads(tx_in_addr) if tx_in_addr else []
                tx_out_addr = json.loads(tx_out_addr) if tx_out_addr else []
            except json.JSONDecodeError:
                tx_in_addr, tx_out_addr = [], []

            transactions.append({
                "timestamp": timestamp,
                "txid": txid,
                "tx_in_addr": tx_in_addr,
                "tx_out_addr": tx_out_addr,
                "total_sent": total_sent,
                "fee_paid": fee_paid,
                "fee_per_vbyte": fee_per_vbyte
            })

        conn.close()
        return transactions


def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    response = requests.post(url, json=payload)
    return response.json()


def fetch_btc_price() -> float:
    """Fetches the current BTC price in USD from CoinGecko."""
    url = 'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        return data['bitcoin']['usd']
    else:
        print(f"Error fetching BTC price: {response.status_code}")
        return None

def address_to_scripthash(address):
    return bitcoinlib.keys.deserialize_address(address)["public_key_hash"]


def fetch_historical_btc_price(timestamp: int) -> float:
    """
    Fetch the historical BTC price in USD closest to the given UNIX timestamp using CoinCap API.
    :param timestamp: UNIX timestamp in seconds
    :return: BTC price in USD or None if not found
    
    url = f"https://api.coincap.io/v2/assets/bitcoin/history"
    
    # CoinCap requires timestamps in milliseconds
    params = {
        "interval": "m5",  # 5-minute intervals for more accuracy
        "start": timestamp * 1000,
        "end": (timestamp + 600) * 1000  # 10 minutes window
    }

    headers = {
        "Authorization": f"Bearer {COINCAP_API_KEY}"
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        if data["data"]:
            # Return the closest price available
            return float(data["data"][0]["priceUsd"])
        else:
            print("No historical data available for the given timestamp.")
    else:
        print(f"Failed to fetch historical BTC price. Status code: {response.status_code}")
    """
    return 0.0

#timestamp = 1231006505  # UNIX timestamp for a block (e.g. block_time from blockchain)
#price = fetch_historical_btc_price(timestamp)
#print(f"BTC price at time {timestamp} was ${price}")

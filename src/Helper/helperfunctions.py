import hashlib
import codecs
import bech32
import requests
import base58
import sqlite3
from datetime import datetime, timedelta
import json
import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
from node_data import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


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
    """Convert Bitcoin address to Electrum's scripthash format"""
    try:
        if address.startswith("1"):  # P2PKH Address
            decoded = base58.b58decode_check(address).hex()
            script = f"76a914{decoded[2:]}88ac"
        elif address.startswith("3"):  # P2SH Address
            decoded = base58.b58decode_check(address).hex()
            script = f"a914{decoded[2:]}87"
        elif address.startswith("bc1"):  # Bech32 SegWit Address
            hrp, data = bech32.bech32_decode(address)
            if not data:
                raise ValueError("Invalid Bech32 address")
            script = codecs.encode(hashlib.sha256(bytes(bech32.convertbits(data[1:], 5, 8, False))).digest(), 'hex').decode()
        else:
            raise ValueError(f"Unsupported address type: {address}")

        scripthash = hashlib.sha256(bytes.fromhex(script)).digest()[::-1].hex()
        return scripthash
    except Exception as e:
        print(f"❌ Address conversion error: {e}")
        return None
import hashlib
import base58
import sqlite3


def address_to_scripthash(address:str) -> str:
    """Convert a Bitcoin address to an Electrum scripthash."""
    decoded = base58.b58decode_check(address)
    pubkey_hash = decoded[1:]
    scripthash = hashlib.sha256(pubkey_hash).digest()[::-1].hex()
    return scripthash


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
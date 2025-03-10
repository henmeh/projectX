import requests
import sqlite3
import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
from Helper.helperfunctions import create_table, store_data, fetch_data
from node_data import RPC_USER, RPC_PASSWORD, RPC_HOST
from datetime import datetime
from bitcoinrpc.authproxy import AuthServiceProxy


class WhaleTracking():

    def __init__(self, db_path: str = "/media/henning/Volume/Programming/projectX/src/mempool_transactions.db"):
        self.db_path = db_path
        self.rpc_user = RPC_USER
        self.rpc_password = RPC_PASSWORD
        self.rpc_host = RPC_HOST

        try:
            create_table(self.db_path, """CREATE TABLE IF NOT EXISTS whale_wallets (address TEXT PRIMARY KEY, last_balance REAL, last_checked TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
            create_table(self.db_path, """CREATE TABLE IF NOT EXISTS whale_wallet_history (id INTEGER PRIMARY KEY AUTOINCREMENT, address TEXT, balance REAL, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (address) REFERENCES whale_wallets(address))""")
        except Exception as e:
            print(f"❌ Database creation filed: {e}")
        
        try:
            self.rpc = AuthServiceProxy(f"http://{self.rpc_user}:{self.rpc_password}@{self.rpc_host}", timeout=180)
            print("✅ RPC Connection Established!")
        except Exception as e:
            print(f"❌ RPC Connection Failed: {e}")
            self.rpc = None


    def fetch_balances(self, whale_addresses: list) -> dict:
        """
        Fetch BTC balances for a list of whale addresses from the local Bitcoin node.
        """
        if not whale_addresses:
            return {}

        try:
            result = self.rpc.scantxoutset("start", [{"desc": f"addr({addr})"} for addr in whale_addresses])
            if "unspents" in result:
                balances = {}
                for utxo in result["unspents"]:
                    addr = utxo["address"]
                    balances[addr] = balances.get(addr, 0) + utxo["amount"]
                return balances
        except Exception as e:
            print(f"❌ Error fetching balances: {e}")
        
        return {}
    

    def track_wallet_balance(self, address):
        balance = self.fetch_balances(address)
        if balance is None:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if address already exists
        cursor.execute("SELECT last_balance FROM whale_wallets WHERE address = ?", (address,))
        result = cursor.fetchone()

        if result:
            last_balance = result[0]
            if balance != last_balance:
                # Update latest balance
                cursor.execute("UPDATE whale_wallets SET last_balance = ?, last_checked = ? WHERE address = ?",
                            (balance, datetime.utcnow(), address))
                # Store historical record
                cursor.execute("INSERT INTO whale_wallet_history (address, balance) VALUES (?, ?)", (address, balance))
        else:
            # Insert new whale wallet
            cursor.execute("INSERT INTO whale_wallets (address, last_balance) VALUES (?, ?)", (address, balance))
            cursor.execute("INSERT INTO whale_wallet_history (address, balance) VALUES (?, ?)", (address, balance))

        conn.commit()
        conn.close()
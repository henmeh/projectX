import json
import sqlite3
import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
from Helper.helperfunctions import create_table, fetch_btc_price, store_data
from Mempool.mempool import Mempool
from node_data import RPC_USER, RPC_PASSWORD, RPC_HOST
from datetime import datetime
from NodeConnect.node_connect import NodeConnect
from whale_alert import WhaleAlerts

class WhaleTracking():

    def __init__(self, db_path: str = "/media/henning/Volume/Programming/projectX/src/mempool_transactions.db"):
        self.db_path = db_path
        self.rpc_user = RPC_USER
        self.rpc_password = RPC_PASSWORD
        self.rpc_host = RPC_HOST
        self.mempool = Mempool()
        self.whale_alert = WhaleAlerts()

        try:
            create_table(self.db_path, """CREATE TABLE IF NOT EXISTS whale_wallets (address TEXT PRIMARY KEY, last_balance REAL, last_checked TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
            create_table(self.db_path, """CREATE TABLE IF NOT EXISTS whale_wallet_history (id INTEGER PRIMARY KEY AUTOINCREMENT, address TEXT, balance REAL, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (address) REFERENCES whale_wallets(address))""")
        except Exception as e:
            print(f"❌ Database creation filed: {e}")
        
        self.node = NodeConnect()


    def fetch_balances(self, whale_addresses: list) -> dict:
        """
        Fetch BTC balances for a list of whale addresses from the local Bitcoin node.
        """
        if not whale_addresses:
            return {}

        try:
            result = self.node.rpc.scantxoutset("start", [{"desc": f"addr({addr})"} for addr in whale_addresses])
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
                            (balance, datetime.now(), address))
                # Store historical record
                cursor.execute("INSERT INTO whale_wallet_history (address, balance) VALUES (?, ?)", (address, balance))
        else:
            # Insert new whale wallet
            cursor.execute("INSERT INTO whale_wallets (address, last_balance) VALUES (?, ?)", (address, balance))
            cursor.execute("INSERT INTO whale_wallet_history (address, balance) VALUES (?, ?)", (address, balance))

        conn.commit()
        conn.close()
    

    def process_tx_batch(self, txids: list, threshold: int, db_path: str, btc_price: float) -> None:
        """Processes a batch of transactions and stores results in the database."""
        tx_data = self.node.rpc_batch_call("getrawtransaction", txids)
        whale_tx = []
        for tx in tx_data:
            sum_btc_sent = sum([out["value"] for out in tx["vout"]])
            sum_btc_input = 0
            if sum_btc_sent > threshold:
                vin_tx_addr = []
                vout_tx_addr = []
                for vin in tx["vin"]:
                    vin_tx = self.node.rpc_call("getrawtransaction", [vin["txid"], True])["result"]
                    vin_out = vin_tx["vout"][vin["vout"]]
                    sum_btc_input += float(vin_out["value"])
                    fee_paid = (float(sum_btc_input) - float(sum_btc_sent)) * 100000000
                    fee_per_vbyte = fee_paid / tx["vsize"]

                    vin_tx_addr.append(vin_tx["vout"][vin["vout"]]["scriptPubKey"]["address"])
                
                for vout in tx["vout"]:
                    if "address" in vout["scriptPubKey"]:
                        vout_tx_addr.append(vout["scriptPubKey"]["address"])
                
                whale_tx.append({
                    "txid": tx["txid"],
                    "size": tx["size"],
                    "vsize": tx["vsize"],
                    "weight": tx["weight"],
                    "tx_in_addr": json.dumps(vin_tx_addr),
                    "tx_out_addr": json.dumps(vout_tx_addr),
                    "fee_paid": fee_paid,
                    "fee_per_vbyte": fee_per_vbyte,
                    "total_sent": float(sum_btc_sent)
                })

                if sum_btc_sent >= self.whale_alert.get_alert_threshold():
                    self.whale_alert.detect_unusual_activity({"sum_btc_sent": sum_btc_sent, "tx_in_addr": vin_tx_addr, "tx_out_addr": vout_tx_addr, "txid": tx["txid"]})

        for tx in whale_tx:
            store_data(db_path, "INSERT INTO mempool_transactions (txid, size, vsize, weight, tx_in_addr, tx_out_addr, fee_paid, fee_per_vbyte, total_sent, btcusd) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (tx["txid"], tx["size"], tx["vsize"], tx["weight"], tx["tx_in_addr"], tx["tx_out_addr"], tx["fee_paid"], tx["fee_per_vbyte"], tx["total_sent"], btc_price))


    def get_whale_transactions(self, threshold: int=10, batch_size=25)-> list:
        """Fetches transactions from the mempool that are above the threshold."""
        mempool_txids = self.mempool.get_mempool_txids()
        btc_price = fetch_btc_price()
        
        create_table(self.db_path, '''CREATE TABLE IF NOT EXISTS mempool_transactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        txid TEXT,
                        size INTEGER,
                        vsize INTEGER,
                        weight INTEGER,
                        tx_in_addr ARRAY,
                        tx_out_addr ARRAY,
                        fee_paid REAL,
                        fee_per_vbyte REAL,
                        total_sent REAL,
                        btcusd REAL)''')

        for i in range(0, len(mempool_txids), batch_size):
            self.process_tx_batch(mempool_txids[i:i+batch_size], threshold, self.db_path, btc_price)

        return True
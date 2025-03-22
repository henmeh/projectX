import json
import sqlite3
from collections import Counter
import pandas as pd
import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
from Helper.helperfunctions import create_table, fetch_btc_price, store_data, fetch_whale_transactions, address_to_scripthash
from Mempool.mempool import Mempool
from node_data import RPC_USER, RPC_PASSWORD, RPC_HOST
from NodeConnect.node_connect import NodeConnect
from .whale_alert import WhaleAlerts

class WhaleTracking():

    def __init__(self, db_path: str = "/media/henning/Volume/Programming/projectX/src/mempool_transactions.db", days=7):
        self.db_path = db_path
        self.days = days
        self.rpc_user = RPC_USER
        self.rpc_password = RPC_PASSWORD
        self.rpc_host = RPC_HOST
        self.mempool = Mempool()
        self.whale_alert = WhaleAlerts()
        try:
            #create_table(self.db_path, """CREATE TABLE IF NOT EXISTS whale_wallets (address TEXT PRIMARY KEY, last_balance REAL, last_checked TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
            #create_table(self.db_path, """CREATE TABLE IF NOT EXISTS whale_wallet_history (id INTEGER PRIMARY KEY AUTOINCREMENT, address TEXT, balance REAL, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (address) REFERENCES whale_wallets(address))""")
            create_table(self.db_path, """CREATE TABLE IF NOT EXISTS whale_balance_history (id INTEGER PRIMARY KEY AUTOINCREMENT, address TEXT NOT NULL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, confirmed_balance REAL NOT NULL, unconfirmed_balance REAL NOT NULL)""")
        except Exception as e:
            print(f"❌ Database creation filed: {e}")
        try:
            self.whale_transactions = fetch_whale_transactions(self.db_path, self.days)
        except Exception as e:
            print(f"❌ RPC Connection Failed: {e}")
            self.whale_transactions = []
        self.node = NodeConnect()


    def whale_behavior_patterns(self) -> list:
        """
        Analysing Whale transactions for recurring patterns
        """
        # Convert to DataFrame
        df = pd.DataFrame(self.whale_transactions)

        # Ensure timestamps are in datetime format
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Extract all input addresses
        all_inputs = []
        for addresses in df["tx_in_addr"]:
            try:
                all_inputs.extend(addresses)
            except:
                continue

        # Extract all output addresses
        all_outputs = []
        for addresses in df["tx_out_addr"]:
            try:
                all_outputs.extend(addresses)  
            except:
                continue

        # Count occurrences of input (sending) addresses
        input_counts = Counter(all_inputs)
        top_senders = input_counts.most_common(10)  # Top 10 frequent senders

        # Count occurrences of output (receiving) addresses
        output_counts = Counter(all_outputs)
        top_receivers = output_counts.most_common(10)  # Top 10 frequent receivers

        # Identify addresses that frequently send & receive BTC
        recurring_addresses = set(input_counts.keys()) & set(output_counts.keys())

        # Analyze Whale Activity by Time
        df["hour"] = df["timestamp"].dt.hour  # Extract hour from timestamp
        whale_activity_by_hour = df["hour"].value_counts().sort_index()

        return top_senders, top_receivers, recurring_addresses, whale_activity_by_hour
  
    
    def fetch_balance(self, address: str) -> dict:
        from bitcoinrpc.authproxy import AuthServiceProxy

        self.rpc_user = RPC_USER
        self.rpc_password = RPC_PASSWORD
        self.rpc_host = RPC_HOST
        try:
            self.rpc = AuthServiceProxy(f"http://{self.rpc_user}:{self.rpc_password}@{self.rpc_host}", timeout=180)
            print("✅ RPC Connection Established!")
        except Exception as e:
            print(f"❌ RPC Connection Failed: {e}")
            self.rpc = None

        """Fetch the balance of a single address"""
        print(f"Fetching balance for address: {address}")
        
        addresses = ["bc1qkh4xr4x8hjra8wymcnyvzzxu6alxeerwkrufln", "bc1qkh4xr4x8hjra8wymcnyvzzxu6alxeerwkrufln", "bc1qkh4xr4x8hjra8wymcnyvzzxu6alxeerwkrufln"]
        # Convert the address to a descriptor
        descriptor = [f"addr({address})" for address in addresses]
        
        # Call the Bitcoin Core API with the descriptor
        try:
            batch = [["scantxoutset", "start", [f"addr({address})" for address in addresses]]]
            balance = self.rpc.batch_(batch)
            return balance
        except Exception as e:
            print(f"Error fetching balance: {e}")
            return None
        

    def fetch_balances(self, addresses: list) -> list:
        """Fetch balances for multiple addresses"""
        return {addr: self.fetch_balance(addr) for addr in addresses}
    

    def store_balance_history(self, address, confirmed, unconfirmed):
        """Store balance history in the database"""
        store_data(self.db_path, "INSERT INTO whale_balance_history (address, confirmed_balance, unconfirmed_balance) VALUES (?, ?, ?)", (address, confirmed, unconfirmed))


    def detect_whale_trends(self, address):
        """Analyze whale balance trends over a given period"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT timestamp, confirmed_balance 
            FROM whale_balance_history 
            WHERE address = ? 
            ORDER BY timestamp ASC
        """
        
        df = pd.read_sql_query(query, conn, params=(address,))
        conn.close()

        if df.empty or len(df) < 2:
            return f"No enough data for {address}"

        # Convert timestamp to pandas datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Calculate trend: Compare first and last recorded balance
        first_balance = df["confirmed_balance"].iloc[0]
        last_balance = df["confirmed_balance"].iloc[-1]

        if last_balance > first_balance:
            return f"✅ Whale {address} is ACCUMULATING ({first_balance} BTC → {last_balance} BTC)"
        elif last_balance < first_balance:
            return f"🚨 Whale {address} is DISTRIBUTING ({first_balance} BTC → {last_balance} BTC)"
        else:
            return f"⚖️ Whale {address} has NO SIGNIFICANT CHANGE ({first_balance} BTC)"
        
    
    def track_whale_balances(self, addresses):
        print("Hallo von track_whale_balances")
        """Fetch and store balances for multiple whales"""
        # Format the addresses into the required descriptor format
        print(addresses)
        balances = self.fetch_balance(addresses)
        print(balances)
        return balances



        #for address in addresses:
        #    balance = self.fetch_balance(address)
        #    print(balance)
        #    #if balance:
        #    #    self.store_balance_history(address, balance["confirmed"], balance["unconfirmed"])
  

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


    def get_whale_transactions(self, threshold: int=100, batch_size=25)-> list:
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
    

    def get_whale_addresses(self, min_tx_count=3):
        """
        Extract whale addresses based on transaction frequency.
        min_tx_count: Number of times an address should appear in whale transactions.
        """
        conn = sqlite3.connect(self.db_path)
        
        # Fetch whale transactions from the database
        df = pd.read_sql_query("SELECT tx_in_addr, tx_out_addr FROM mempool_transactions", conn)
        conn.close()

        all_addresses = []

        # Extract all input and output addresses
        for col in ["tx_in_addr", "tx_out_addr"]:
            df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else [])  # Convert string to list
            for addr_list in df[col]:
                all_addresses.extend(addr_list)

        # Count occurrences of each address
        address_counts = pd.Series(all_addresses).value_counts()

        # Filter addresses that appear at least 'min_tx_count' times
        whale_addresses = address_counts[address_counts >= min_tx_count].index.tolist()
        
        return whale_addresses


    def merge_with_clusters(self, whale_addresses, clustered_addresses):
        """
        Merge whale addresses with clustered addresses.
        This helps track addresses that interact frequently.
        """
        expanded_whale_addresses = set(whale_addresses)

        for cluster in clustered_addresses:
            if any(addr in expanded_whale_addresses for addr in cluster):
                expanded_whale_addresses.update(cluster)

        return list(expanded_whale_addresses)

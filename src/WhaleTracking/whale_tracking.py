import json
import sqlite3
from collections import Counter
import pandas as pd
import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
from Helper.helperfunctions import create_table, fetch_btc_price, store_data, fetch_whale_transactions, get_existing_txids, fetch_data
from Mempool.mempool import Mempool
from .whale_alert import WhaleAlerts

class WhaleTracking():

    def __init__(self, node = None, db_path: str = "/media/henning/Volume/Programming/projectX/src/mempool_transactions.db", days=7):
        self.db_path = db_path
        self.days = days
        try:
            create_table(self.db_path, """CREATE TABLE IF NOT EXISTS whale_balance_history (id INTEGER PRIMARY KEY AUTOINCREMENT, address TEXT NOT NULL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, confirmed_balance REAL NOT NULL, unconfirmed_balance REAL NOT NULL)""")
        except Exception as e:
            print(f"❌ Database creation filed: {e}")
        try:
            self.whale_transactions = fetch_whale_transactions(self.db_path, self.days)
        except Exception as e:
            print(f"❌ RPC Connection Failed: {e}")
            self.whale_transactions = []
        self.node = node


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
  
    
    def fetch_balance(self, addresses: list) -> dict:
        try:
            batch = [["scantxoutset", "start", [f"addr({address})" for address in addresses]]]
            balance = self.node.rpc_batch_call(batch)
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
        whale_alert = WhaleAlerts()
        existing_txids = get_existing_txids(db_path, txids)
        new_txids = [txid for txid in txids if txid not in existing_txids]
    
        if not new_txids:
            return
        
        tx_data = self.node.rpc_batch_call("getrawtransaction", new_txids)
        whale_tx = []
        for tx in tx_data:
            sum_btc_sent = sum([out["value"] for out in tx["vout"]])
            sum_btc_input = 0
            if sum_btc_sent > threshold:
                vin_tx_addr = []
                vout_tx_addr = []

                #for the input transactions the txid and the vout needs to be saved
                vin_txs_data_to_store = {}
                
                for vin in tx["vin"]:
                    vin_txs_data_to_store[vin["txid"]] = vin["vout"]
                
                for vout in tx["vout"]:
                    if "address" in vout:
                        vout_tx_addr.append(vout["address"])
                    if "scriptPubKey" in vout:
                        if "address" in vout["scriptPubKey"]:
                            vout_tx_addr.append(vout["scriptPubKey"]["address"])
                        else:
                            vout_tx_addr.append("")
                    else:
                        vout_tx_addr.append("")

                vin_txs = self.node.rpc_batch_call("getrawtransaction", list(vin_txs_data_to_store.keys()))

                for vin_tx in vin_txs:
                    vin_out = vin_tx["vout"][vin_txs_data_to_store[vin_tx["txid"]]]
                    vin_tx_addr.append(vin_tx["vout"][vin_txs_data_to_store[vin_tx["txid"]]]["scriptPubKey"]["address"])
                    sum_btc_input += float(vin_out["value"])

                fee_paid = (float(sum_btc_input) - float(sum_btc_sent)) * 100000000
                fee_per_vbyte = fee_paid / tx["vsize"]
                                
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

                if sum_btc_sent >= whale_alert.get_alert_threshold():
                    whale_alert.detect_unusual_activity({"sum_btc_sent": sum_btc_sent, "tx_in_addr": vin_tx_addr, "tx_out_addr": vout_tx_addr, "txid": tx["txid"]})

        for tx in whale_tx:
            store_data(db_path, "INSERT INTO mempool_transactions (txid, size, vsize, weight, tx_in_addr, tx_out_addr, fee_paid, fee_per_vbyte, total_sent, btcusd) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (tx["txid"], tx["size"], tx["vsize"], tx["weight"], tx["tx_in_addr"], tx["tx_out_addr"], tx["fee_paid"], tx["fee_per_vbyte"], tx["total_sent"], btc_price))


    def get_whale_transactions(self, threshold: int=100, batch_size=25)-> list:
        """Fetches transactions from the mempool that are above the threshold."""
        mempool = Mempool(self.node)

        mempool_txids = mempool.get_mempool_txids()
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
    

    def get_whale_addresses(self, min_tx_count=3, threshold=500):
        """Fetches unique addresses from whale transactions and filters them."""
        query = f"""
        SELECT DISTINCT tx_in_addr FROM mempool_transactions WHERE total_sent >= {threshold} 
        UNION 
        SELECT DISTINCT tx_out_addr FROM mempool_transactions WHERE total_sent >= {threshold};
        """
        raw_addresses = fetch_data(self.db_path, query)

        # Flatten the list (since tx_in_addr and tx_out_addr are stored as JSON arrays)
        unique_addresses = set()
        for addr_list in raw_addresses:
            addr_list = json.loads(addr_list[0])  # Convert JSON string to list
            unique_addresses.update(addr_list)

        # Filter addresses based on balance
        whale_addresses = []
        balance_data = self.fetch_balances(list(unique_addresses))

        for addr, balance in balance_data.items():
            if balance >= threshold:  # Consider it a whale if balance exceeds threshold
                whale_addresses.append(addr)

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

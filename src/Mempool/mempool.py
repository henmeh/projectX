import socket
from bitcoinrpc.authproxy import AuthServiceProxy
import matplotlib.pyplot as plt
import requests
import json
import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
from node_data import ELECTRUM_HOST, ELECTRUM_PORT, RPC_USER, RPC_PASSWORD, RPC_HOST
from Helper.helperfunctions import create_table, store_data, fetch_data, send_telegram_alert
import datetime


class Mempool():

    def __init__(self):
        self.electrum_host = ELECTRUM_HOST
        self.electrum_port = ELECTRUM_PORT
        self.rpc_user = RPC_USER
        self.rpc_password = RPC_PASSWORD
        self.rpc_host = RPC_HOST
        self.db_mempool_transactions_path = "/media/henning/Volume/Programming/projectX/src/mempool_transactions.db"
        try:
            self.rpc = AuthServiceProxy(f"http://{self.rpc_user}:{self.rpc_password}@{self.rpc_host}", timeout=180)
            print("✅ RPC Connection Established!")
        except Exception as e:
            print(f"❌ RPC Connection Failed: {e}")
            self.rpc = None


    def get_mempool_feerates(self, block_vsize_limit:int=1000000) -> json:
        """Fetching Feerates from local mempool"""
        request_data = {
            "id": 0,
            "method": "mempool.get_fee_histogram",
            "params": []
        }
        try:
            with socket.create_connection((self.electrum_host, self.electrum_port)) as sock:
                sock.sendall(json.dumps(request_data).encode() + b'\n')
                response = sock.recv(4096).decode()
                fee_histogram = json.loads(response)["result"]
                
                create_table(self.db_mempool_transactions_path, '''CREATE TABLE IF NOT EXISTS mempool_fee_histogram (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        histogram TEXT,
                        fast_fee REAL,
                        medium_fee REAL,
                        low_fee REAL)''')
                
                total_vsize = 0
                vsize_25 = block_vsize_limit * 0.25  # Fast: Top 25%
                vsize_50 = block_vsize_limit * 0.50  # Medium: Top 50%
                vsize_75 = block_vsize_limit * 0.75  # Low: Top 75%

                fast_fee = medium_fee = low_fee = None

                # Sum transaction sizes and find percentiles
                for fee_rate, vsize in fee_histogram:
                    total_vsize += vsize

                    if fast_fee is None and total_vsize >= vsize_25:
                        fast_fee = fee_rate
                    if medium_fee is None and total_vsize >= vsize_50:
                        medium_fee = fee_rate
                    if low_fee is None and total_vsize >= vsize_75:
                        low_fee = fee_rate

                    if total_vsize >= block_vsize_limit:
                        break

                store_data(self.db_mempool_transactions_path, "INSERT INTO mempool_fee_histogram (histogram, fast_fee, medium_fee, low_fee) VALUES (?, ?, ?, ?)", (json.dumps(fee_histogram), fast_fee, medium_fee, low_fee))
                
                return {
                    "fast": fast_fee,
                    "medium": medium_fee,
                    "low": low_fee
                }        
        except Exception as e:
            return f"Error: {str(e)}"
    

    def plot_fee_histogram_actual(self):
        """Fetches and plots the mempool fee histogram."""
        fee_histogram_actual = fetch_data(self.db_mempool_transactions_path, "SELECT histogram FROM mempool_fee_histogram ORDER BY timestamp DESC LIMIT 1")[0][0]
        fee_histogram_actual_list = json.loads(fee_histogram_actual)
        
        # Extract fee rates and total vsize
        fee_rates = [entry[0] for entry in fee_histogram_actual_list]
        vsizes = [entry[1] for entry in fee_histogram_actual_list]

        # Plot the histogram
        plt.figure(figsize=(10, 6))
        plt.bar(fee_rates, vsizes, width=1.5, color="blue", alpha=0.7)

        plt.xlabel("Fee Rate (sats/vB)")
        plt.ylabel("Total Vsize (vB)")
        plt.title("Mempool Fee Rate Distribution")
        plt.yscale("log")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        
        plt.show()
    

    def get_mempool_stats(self) -> tuple:
        """Fetches the mempool size and transaction count from Electrum server."""
        try:
            result = self.electrum_request("mempool.get_fee_histogram")
            if "result" in result:
                fee_histogram = result["result"]
                total_mempool_size = sum([fee[1] for fee in fee_histogram])
                total_tx_count = len(fee_histogram)
                
                return total_mempool_size, total_tx_count
            else:
                print("Error: No result in response")
                return None
        except Exception as e:
            print(f"Error fetching mempool stats: {e}")
            return None
    

    def electrum_request(self, method: str, params=[])-> json:
        """Sends a JSON-RPC request to the Electrum server."""
        request_data = json.dumps({"id": 0, "method": method, "params": params}) + "\n"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.electrum_host, self.electrum_port))
            s.sendall(request_data.encode("utf-8"))
            response = s.recv(8192).decode("utf-8")
        return json.loads(response)
    

    def rpc_call(self, method: str, params=[]) -> json:
        """Helper function to call Bitcoin Core RPC"""
        payload = {"jsonrpc": "1.0", "id": method, "method": method, "params": params}
        response = requests.post(f"http://{self.rpc_user}:{self.rpc_password}@{self.rpc_host}/", json=payload)
        return response.json()
    

    def rpc_batch_call(self, method: str, params: list) -> json:
        """Helper function to call Bitcoin Core RPC with batch requests"""
        batch = [[method, param, True] for param in params]
        try:
            responses = self.rpc.batch_(batch)
            return responses
        except Exception as e:
            print(f"RPC Error: {e}")
            return []


    def get_transaction_fee(self, txid):
        """Fetches the fee and fee rate of a transaction in the mempool."""
        try:
            tx = self.rpc_call("getrawtransaction", [txid, True])
            if "error" in tx and tx["error"]:
                print(f"Skipping {txid}: {tx['error']['message']}")
                return None

            vsize = tx["result"]["vsize"]

            output_sum = sum([out["value"] for out in tx["result"]["vout"]]) * 1e8  

            input_sum = 0
            for vin in tx["result"]["vin"]:
                if "txid" in vin and "vout" in vin:
                    prev_tx = self.rpc_call("getrawtransaction", [vin["txid"], True])
                    if "error" in prev_tx and prev_tx["error"]:
                        print(f"Skipping input {vin['txid']}: {prev_tx['error']['message']}")
                        continue
                    input_sum += prev_tx["result"]["vout"][vin["vout"]]["value"] * 1e8  

            fee = input_sum - output_sum
            fee_rate = fee / vsize if vsize > 0 else 0 

            return {"txid": txid, "fee": fee, "fee_rate": fee_rate}
        
        except Exception as e:
            print(f"Error fetching fee for {txid}: {e}")
            return None
  
    
    def process_tx_batch(self, txids: list, threshold: int, alert_threshold: int, db_path: str, btc_price: float) -> None:
        """Processes a batch of transactions and stores results in the database."""
        tx_data = self.rpc_batch_call("getrawtransaction", txids)
        whale_tx = []
        for tx in tx_data:
            sum_btc_sent = sum([out["value"] for out in tx["vout"]])
            sum_btc_input = 0
            if sum_btc_sent > threshold:
                vin_tx_addr = []
                vout_tx_addr = []
                for vin in tx["vin"]:
                    vin_tx = self.rpc_call("getrawtransaction", [vin["txid"], True])["result"]
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

                if sum_btc_sent >= alert_threshold:
                    # Construct alert message
                    message = (
                        f"🚨 *Whale Alert!* 🚨\n"
                        f"💰 *{sum_btc_sent} BTC* transferred!\n"
                        f"📥 *From:* {', '.join(vin_tx_addr[:3])}...\n"
                        f"📤 *To:* {', '.join(vout_tx_addr[:3])}...\n"
                        f"⏳ *Time:* {datetime.datetime.now()}\n"
                        f"🔗 [View Transaction](https://mempool.space/tx/{tx['txid']})"
                    )

                    # Send alert
                    send_telegram_alert(message)

        for tx in whale_tx:
            store_data(db_path, "INSERT INTO mempool_transactions (txid, size, vsize, weight, tx_in_addr, tx_out_addr, fee_paid, fee_per_vbyte, total_sent, btcusd) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (tx["txid"], tx["size"], tx["vsize"], tx["weight"], tx["tx_in_addr"], tx["tx_out_addr"], tx["fee_paid"], tx["fee_per_vbyte"], tx["total_sent"], btc_price))
        

    def get_mempool_txids(self)-> list:
        """Fetches transaction IDs from the mempool."""
        try:
            response = self.rpc_call("getrawmempool")
            mempool_transaction_ids = list(response["result"])
            return mempool_transaction_ids 
        except Exception as e:
            print(f"Error fetching mempool txids: {e}")
            return []
    

    def get_whale_transactions(self, threshold: int=10, alert_threshold: int=100, batch_size=25)-> list:
        """Fetches transactions from the mempool that are above the threshold."""
        mempool_txids = self.get_mempool_txids()
        btc_price = self.fetch_btc_price()
        
        create_table(self.db_mempool_transactions_path, '''CREATE TABLE IF NOT EXISTS mempool_transactions (
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
            self.process_tx_batch(mempool_txids[i:i+batch_size], threshold, alert_threshold, self.db_mempool_transactions_path, btc_price)

        return True


    def fetch_btc_price(self) -> float:
        """Fetches the current BTC price in USD from CoinGecko."""
        url = 'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd'
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            return data['bitcoin']['usd']
        else:
            print(f"Error fetching BTC price: {response.status_code}")
            return None
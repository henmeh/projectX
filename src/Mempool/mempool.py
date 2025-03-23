import socket
import matplotlib.pyplot as plt
import json
import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
from Helper.helperfunctions import create_table, store_data, fetch_data

class Mempool():

    def __init__(self, node):
        self.db_mempool_transactions_path = "/media/henning/Volume/Programming/projectX/src/mempool_transactions.db"
        self.node = node


    def get_mempool_feerates(self, block_vsize_limit:int=1000000) -> json:
        """Fetching Feerates from local mempool"""
        request_data = {
            "id": 0,
            "method": "mempool.get_fee_histogram",
            "params": []
        }
        try:
            with socket.create_connection((self.node.get_node_data()["electrum_host"], self.node.get_node_data()["electrum_port"])) as sock:
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
            result = self.node.electrum_request("mempool.get_fee_histogram")
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
 
    
    def get_transaction_fee(self, txid):
        """Fetches the fee and fee rate of a transaction in the mempool."""
        try:
            tx = self.node.rpc_call("getrawtransaction", [txid, True])
            if "error" in tx and tx["error"]:
                print(f"Skipping {txid}: {tx['error']['message']}")
                return None

            vsize = tx["result"]["vsize"]

            output_sum = sum([out["value"] for out in tx["result"]["vout"]]) * 1e8  

            input_sum = 0
            for vin in tx["result"]["vin"]:
                if "txid" in vin and "vout" in vin:
                    prev_tx = self.node.rpc_call("getrawtransaction", [vin["txid"], True])
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
       

    def get_mempool_txids(self)-> list:
        """Fetches transaction IDs from the mempool."""
        try:
            response = self.node.rpc_call("getrawmempool")
            mempool_transaction_ids = list(response["result"])
            return mempool_transaction_ids
        except Exception as e:
            print(f"Error fetching mempool txids: {e}")
            return []
    
import socket
import pandas as pd
import multiprocessing
from bitcoinrpc.authproxy import AuthServiceProxy
import requests
import math
import json
from Helper.helperfunctions import create_table, store_data
from node_data import ELECTRUM_HOST, ELECTRUM_PORT, RPC_USER, RPC_PASSWORD, RPC_HOST


class Mempool():

    def __init__(self):
        self.electrum_host = ELECTRUM_HOST
        self.electrum_port = ELECTRUM_PORT
        self.rpc_user = RPC_USER
        self.rpc_password = RPC_PASSWORD
        self.rpc_host = RPC_HOST
        self.db_mempool_transactions_path = "/media/henning/Volume/Programming/projectX/src/mempol_transactions.db"
        try:
            self.rpc = AuthServiceProxy(f"http://{self.rpc_user}:{self.rpc_password}@{self.rpc_host}", timeout=180)
            print("✅ RPC Connection Established!")
        except Exception as e:
            print(f"❌ RPC Connection Failed: {e}")
            self.rpc = None


    def get_mempool_feerates(self) -> json:
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
                fee_rates = []
                for fee_rate, vsize in json.loads(response)["result"]:
                    if 1 < fee_rate < 50:
                        weight = int(math.sqrt(vsize))
                        fee_rates.extend([fee_rate] * weight)
                fee_rates.sort(reverse=True)
                return fee_rates            
        except Exception as e:
            return f"Error: {str(e)}"
    

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

    def process_tx_batch(self, txids, threshold, db_path):
        """Processes a batch of transactions and stores results in the database."""
        tx_data = self.rpc_batch_call("getrawtransaction", txids)
        for tx in tx_data:
            sum_btc_sent = sum([out["value"] for out in tx["vout"]])
            if sum_btc_sent > threshold:
                store_data(db_path, "INSERT INTO mempool_transactions (txid, size, vsize, weight, num_tx_in, num_tx_out, total_sent) VALUES (?, ?, ?, ?, ?, ?, ?)", (tx["txid"], tx["size"], tx["vsize"], tx["weight"], len(tx["vin"]), len(tx["vout"]), float(sum_btc_sent)))
    

    def get_mempool_txids(self)-> list:
        """Fetches transaction IDs from the mempool."""
        try:
            response = self.rpc_call("getrawmempool")
            mempool_transaction_ids = list(response["result"])
            return mempool_transaction_ids 
        except Exception as e:
            print(f"Error fetching mempool txids: {e}")
            return []
    

    def get_whale_transactions(self, threshold: int=10, batch_size=25)-> list:
        """Fetches transactions from the mempool that are above the threshold."""
        mempool_txids = self.get_mempool_txids()
        
        create_table(self.db_mempool_transactions_path, '''CREATE TABLE IF NOT EXISTS mempool_transactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        txid TEXT,
                        size INTEGER,
                        vsize INTEGER,
                        weight INTEGER,
                        num_tx_in INTEGER,
                        num_tx_out INTEGER,
                        total_sent REAL)''')

        for i in range(0, len(mempool_txids), batch_size):
            self.process_tx_batch(mempool_txids[i:i+batch_size], threshold, self.db_mempool_transactions_path)
        
        """
        num_workers = 4
        txid_chunks = [mempool_txids[i:i+batch_size] for i in range(0, len(mempool_txids), batch_size)]

        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.map(self.process_tx_batch(mempool_txids, threshold, self.db_mempool_transactions_path), [(chunk, threshold, self.db_mempool_transactions_path) for chunk in txid_chunks])
        """

        return True
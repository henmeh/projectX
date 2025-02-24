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
        rpc = AuthServiceProxy(f"http://{self.rpc_user}:{self.rpc_password}@{self.rpc_host}")
        batch = [[method, params, True] for params in params]
        return rpc.batch_(batch)
    

    def get_mempool_txids(self)-> list:
        """Fetches transaction IDs from the mempool."""
        response = self.rpc_call("getrawmempool")
        mempool_transaction_ids = list(response["result"])
        return mempool_transaction_ids
    

    def get_whale_transactions(self, threshold: int = 2)-> list:
        """Fetches transactions from the mempool that are above the threshold."""
        mempool_txids = self.get_mempool_txids()
        DB_PATH = "/media/henning/Volume/Programming/projectX/src/mempol_transactions.db"

        create_table(DB_PATH, '''CREATE TABLE IF NOT EXISTS mempool_transactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        txid TEXT,
                        total_sent REAL)''')
    
        for i in range(0, len(mempool_txids), 100):
            txids = mempool_txids[i:i+100]
            tx_data = self.rpc_batch_call("getrawtransaction", [txid for txid in txids])
            for tx in tx_data:
                sum_btc_sent = sum([out["value"] for out in tx["vout"]])
                if sum_btc_sent > threshold:
                    store_data(DB_PATH, "INSERT INTO mempool_transactions (txid, total_sent) VALUES (?, ?)", (tx["txid"], float(sum_btc_sent)))

        return True
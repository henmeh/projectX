import socket
import math
import json
from Helper.helperfunctions import address_to_scripthash
from node_data import ELECTRUM_HOST, ELECTRUM_PORT


class Mempool():

    def __init__(self):
        self.electrum_host = ELECTRUM_HOST
        self.electrum_port = ELECTRUM_PORT


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

    """
    def get_mempool_for_address(self, address: str) -> json:
        scripthash = address_to_scripthash(address)
        
        request_data = {
            "id": 0,
            "method": "blockchain.scripthash.get_mempool",
            "params": [scripthash]
        }

        try:
            with socket.create_connection((self.electrum_host, self.electrum_port)) as sock:
                sock.sendall(json.dumps(request_data).encode() + b'\n')
                response = sock.recv(4096).decode()
                return json.loads(response)
        
        except Exception as e:
            return f"Error: {str(e)}"
    

    def get_mempool_fees(self) -> list:
        headers = {"Content-Type": "application/json"}
        payload = json.dumps({"jsonrpc": "2.0", "id": 0, "method": "getrawmempool", "params": [True]})

        response = requests.post(self.electrum_host, auth=(BITCOIN_RPC_USER, BITCOIN_RPC_PASSWORD), headers=headers, data=payload)
        mempool_data = response.json()["result"]

        fee_rates = []

        for txid, tx_data in mempool_data.items():
            fee = tx_data["fees"]["base"]  # Fee in BTC
            vsize = tx_data["vsize"]  # Virtual size in vBytes
            fee_rate = (fee * 1e8) / vsize  # Convert BTC to sat and divide by vBytes
            fee_rates.append(fee_rate)

        return fee_rates
    """
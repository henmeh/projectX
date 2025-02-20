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
                    print(f"Fee rate: {fee_rate} sat/vB, Vsize: {vsize} vB")
                    if 1 < fee_rate < 50:
                        weight = int(math.sqrt(vsize))
                        fee_rates.extend([fee_rate] * weight)
                        #fee_rates.extend([fee_rate] * int(math.log(vsize + 1)))
                        #fee_rates.append(fee_rate * min(vsize, 0.1))
                fee_rates.sort(reverse=True)
                return fee_rates
            
        except Exception as e:
            return f"Error: {str(e)}"
    

    def get_mempool_for_address(self, address: str) -> json:
        """Fetching Mempool data for a specific address"""
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
        """Fetching fee rates from Bitcoin Core mempool"""
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
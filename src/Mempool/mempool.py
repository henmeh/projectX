import socket
import sys
import json
from Helper.helperfunctions import address_to_scripthash
from node_data import ELECTRUM_HOST, ELECTRUM_PORT


class Mempool():

    def __init__(self):
        self.electrum_host = ELECTRUM_HOST
        self.electrum_port = ELECTRUM_PORT


    def get_mempool(self) -> json:
        """Fetching Fee Historgram from local mempool"""
        request_data = {
            "id": 0,
            "method": "mempool.get_fee_histogram",
            "params": []
        }

        try:
            with socket.create_connection((self.electrum_host, self.electrum_port)) as sock:
                sock.sendall(json.dumps(request_data).encode() + b'\n')
                response = sock.recv(4096).decode()
                return json.loads(response)
        
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
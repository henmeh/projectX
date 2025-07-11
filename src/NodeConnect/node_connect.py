from bitcoinrpc.authproxy import AuthServiceProxy
import json
import socket
import requests
import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
from node_data import ELECTRUM_HOST, ELECTRUM_PORT

class NodeConnect():

    def __init__(self, rpc_user, rpc_password, rpc_host, electrum_host=ELECTRUM_HOST, electrum_port=ELECTRUM_PORT):
        self.electrum_host = electrum_host
        self.electrum_port = electrum_port
        self.rpc_user = rpc_user
        self.rpc_password = rpc_password
        self.rpc_host = rpc_host
        try:
            self.rpc = AuthServiceProxy(f"http://{self.rpc_user}:{self.rpc_password}@{self.rpc_host}", timeout=180)
            print("✅ RPC Connection Established!")
        except Exception as e:
            print(f"❌ RPC Connection Failed: {e}")
            self.rpc = None
    

    def get_node(self):
        return self
        
    
    def get_node_data(self) -> dict:
        return ({"electrum_host": self.electrum_host, "electrum_port": self.electrum_port, "rpc_user": self.rpc_user, "rpc_password": self.rpc_password, "rpc_host": self.rpc_host})


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


    def rpc_batch_call(self, method: str, params: list) -> list:
        """Helper function to call Bitcoin Core RPC with batch requests"""
        batch = [[method, param, True] for param in params]
        try:
            responses = self.rpc.batch_(batch)
            return responses
        except Exception as e:
            responses = []
            for param in params:
                response = self.rpc_call(method, [param])
                if response["result"] is not None:
                    responses.append(response)
                else:
                    print(f"RPC Error: {e} for txid {param}")
            return responses
        

        
import requests
import json

# Configure your RPC credentials
RPC_USER = "henning"
RPC_PASSWORD = "test"
RPC_HOST = "127.0.0.1"
RPC_PORT = "8332"

def rpc_call(method, params=[]):
    url = f"http://{RPC_HOST}:{RPC_PORT}/"
    headers = {"content-type": "application/json"}
    payload = json.dumps({"jsonrpc": "1.0", "id": "test", "method": method, "params": params})
    
    try:
        response = requests.post(url, headers=headers, data=payload, auth=(RPC_USER, RPC_PASSWORD))
        if response.status_code == 200:
            return response.json()["result"]
        else:
            print(f"Error: Received non-200 response: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"RPC Connection Failed: {e}")
        return None

if __name__ == "__main__":
    block_height = rpc_call("getrawtransaction", ["920075b0d923da74e23ae810de37b17013aadebfeddb00235d930a05f31ceb1a", True])
    if block_height is not None:
        print(f"✅ Connected! Current block height: {block_height}")
    else:
        print("❌ Failed to connect to Bitcoin node.")

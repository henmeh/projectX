from bitcoinrpc.authproxy import AuthServiceProxy
from node_data import RPC_USER, RPC_PASSWORD, RPC_PORT, RASPBERRY_PI_IP


class Chain():
    def __init__(self):
        self.rpc_url = f"http://{RPC_USER}:{RPC_PASSWORD}@{RASPBERRY_PI_IP}:{RPC_PORT}"
        self.rpc_connection = AuthServiceProxy(self.rpc_url)


    def get_block_height(self) -> int:
        """Fetches the current block height from the Bitcoin Core node."""
        block_height = self.rpc_connection.getblockcount()
        return block_height
from bitcoinrpc.authproxy import AuthServiceProxy
import requests
from Mempool.mempool import Mempool

# Replace with your Raspberry Pi's actual local IP
RASPBERRY_PI_IP = "192.168.178.168"  # Update this
RPC_USER = "raspibolt"  # Use the same username from bitcoin.conf
RPC_PASSWORD = "98k0hhshfjfc1gm"  # Use the same password from bitcoin.conf
RPC_PORT = "8332"

# Connect to Bitcoin Core
#rpc_url = f"http://{RPC_USER}:{RPC_PASSWORD}@{RASPBERRY_PI_IP}:{RPC_PORT}"
#rpc_connection = AuthServiceProxy(rpc_url)

# Test connection by fetching block height
#block_height = rpc_connection.getblockcount()
#print(f"Connected! Current block height: {block_height}")

#mempool_info = rpc_connection.getmempoolinfo()
#print(mempool_info)

#mempool_txs = rpc_connection.getrawmempool()
#print(mempool_txs[:10])  # Print only first 10 TXs to avoid flooding the terminal

#txid = mempool_txs[0]  # Take the first transaction in mempool
#tx_info = rpc_connection.getmempoolentry(txid)
#print(tx_info)


mempool = Mempool()

# Fetch mempool data
mempool_data = mempool.get_mempool()
print(mempool_data)
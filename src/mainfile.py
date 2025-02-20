from bitcoinrpc.authproxy import AuthServiceProxy
import requests
from Mempool.mempool import Mempool

# Replace with your Raspberry Pi's actual local IP


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
fee_rates = mempool.get_mempool_feerates()
fast_fee = fee_rates[int(len(fee_rates) * 0.25)] if len(fee_rates) > 10 else max(fee_rates)
medium_fee = fee_rates[int(len(fee_rates) * 0.5)] if len(fee_rates) > 2 else fee_rates[0]  # Median
low_fee = fee_rates[-1]  # Lowest

print(f"🚀 Fast Fee: {fast_fee:.2f} sat/vB")
print(f"⏳ Medium Fee: {medium_fee:.2f} sat/vB")
print(f"💤 Low Fee: {low_fee:.2f} sat/vB")
import requests
from Mempool.mempool import Mempool
import json

# Replace with your Raspberry Pi's actual local IP


# Connect to Bitcoin Core




#mempool_info = rpc_connection.getmempoolinfo()
#print(mempool_info)

#mempool_txs = rpc_connection.getrawmempool()
#print(mempool_txs[:10])  # Print only first 10 TXs to avoid flooding the terminal

#txid = mempool_txs[0]  # Take the first transaction in mempool
#tx_info = rpc_connection.getmempoolentry(txid)
#print(tx_info)



# Fetch mempool data
#fee_rates = mempool.get_mempool_feerates()
#fast_fee = fee_rates[int(len(fee_rates) * 0.25)] if len(fee_rates) > 10 else max(fee_rates)
#medium_fee = fee_rates[int(len(fee_rates) * 0.5)] if len(fee_rates) > 2 else fee_rates[0]  # Median
#low_fee = fee_rates[-1]  # Lowest

#print(f"🚀 Fast Fee: {fast_fee:.2f} sat/vB")
#print(f"⏳ Medium Fee: {medium_fee:.2f} sat/vB")
#print(f"💤 Low Fee: {low_fee:.2f} sat/vB")

#mempool_size, tx_count = mempool.get_mempool_stats()
#print(mempool_size)
#print(tx_count)

#test = mempool.get_whale_transactions()
#print(test)


from bitcoinrpc.authproxy import AuthServiceProxy
import time
from node_data import RPC_USER, RPC_PASSWORD, RPC_HOST


# Connect with long timeout

from bitcoinrpc.authproxy import AuthServiceProxy
import multiprocessing
import time

# Bitcoin RPC connection
rpc = AuthServiceProxy(f"http://{RPC_USER}:{RPC_PASSWORD}@{RPC_HOST}", timeout=300)

def fetch_transactions(txid_chunk):
    """Fetch raw transactions for a chunk of txids using batch call."""
    batch = [["getrawtransaction", txid, True] for txid in txid_chunk]
    try:
        return rpc.batch_(batch)
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def parallel_fetch_mempool_data(txids, num_workers=4, chunk_size=50):
    """Fetch mempool transactions in parallel."""
    start_time = time.time()

    # Split into chunks
    chunks = [txids[i:i + chunk_size] for i in range(0, len(txids), chunk_size)]

    # Use multiprocessing to fetch in parallel
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(fetch_transactions, chunks)

    # Flatten results (remove None values)
    tx_data = [tx for batch in results if batch for tx in batch]

    print(f"✅ Fetched {len(tx_data)} transactions in {time.time() - start_time:.2f} sec")
    return tx_data

# Run the parallel fetch
#test = parallel_fetch_mempool_data(mempool_txids, num_workers=4, chunk_size=150)

if __name__ == "__main__":
    # Get mempool txids
    mempool = Mempool()
    #mempool_txids = mempool.get_mempool_txids()
    #parallel_fetch_mempool_data(mempool_txids, num_workers=4, chunk_size=150)
    mempool.get_whale_transactions()

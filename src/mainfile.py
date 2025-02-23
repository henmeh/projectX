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


mempool = Mempool()

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

test = mempool.get_whale_transactions()
print(test)
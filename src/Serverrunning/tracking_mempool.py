"""
1. Tracking all mempool transactions that sending more than 100 btc. Tracking is done every minute
2. Tracking feerates every 60 seconds
"""
import time
import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')

from Mempool.mempool import Mempool
from WhaleTracking.whale_tracking import WhaleTracking
from NodeConnect.node_connect import NodeConnect
from node_data import RPC_USER_RASPI, RPC_PASSWORD_RASPI, RPC_HOST_RASPI



if __name__ == "__main__":
        raspi = NodeConnect(RPC_USER_RASPI, RPC_PASSWORD_RASPI, RPC_HOST_RASPI).get_node()
        whaletracking = WhaleTracking(node=raspi)
        mempool = Mempool(node=raspi)
        
        while True:
            #whaletracking.delete_mined_mempool_transactions()
            #whaletracking.process_mempool_transactions()
            mempool.get_mempool_feerates()
           
            print(f"✅ Data stored at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            time.sleep(60)
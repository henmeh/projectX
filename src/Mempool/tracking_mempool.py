import time
import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
from Mempool.mempool import Mempool
from WhaleTracking.whale_tracking import WhaleTracking


if __name__ == "__main__":
        mempool = Mempool()
        whaletracking = WhaleTracking()
        while True:
            whaletracking.get_whale_transactions()
            mempool.get_mempool_feerates()
            print(f"✅ Data stored at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            time.sleep(60)  # Run every 60 seconds

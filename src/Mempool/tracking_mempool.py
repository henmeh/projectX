import time
from mempool import Mempool


if __name__ == "__main__":
        mempool = Mempool()
        while True:
            mempool.get_whale_transactions()
            mempool.get_mempool_feerates()
            print(f"✅ Data stored at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            time.sleep(60)  # Run every 60 seconds

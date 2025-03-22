"""
Fetches all Transactions with btc > 100 from the last days and try to cluster addresses. Tracking is every 24h
"""
import time
import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
from Statistics.address_clustering import AddressClustering

if __name__ == "__main__":

    clustering = AddressClustering()

    while True:
        address_clusters = clustering.run_clustering()
        clustering.store_clusters(address_clusters)

        print(f"✅ Data clustered {time.strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(60*60*24)  # Run once a day
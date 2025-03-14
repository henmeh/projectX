"""
Tracking specific whale wallets
"""  
import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
from WhaleTracking.whale_tracking import WhaleTracking
from Statistics.address_clustering import AddressClustering

if __name__ == "__main__":
   
    print("los gehts")
    whale_walett_tracking = WhaleTracking()
    clustering = AddressClustering()

    print("1")
    whale_addresses = whale_walett_tracking.get_whale_addresses()
    clustered_addresses = clustering.run_clustering()
    whale_addresses_merged = whale_walett_tracking.merge_with_clusters(whale_addresses, clustered_addresses)

    print("2")
    whale_walett_tracking.track_whale_balances(whale_addresses_merged)

    print("3")
    for address in whale_addresses_merged:
        print(address)
        whale_walett_tracking.detect_whale_trends(address)

    print ("Ende")
   
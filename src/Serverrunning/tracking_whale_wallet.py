"""
Tracking specific whale wallets
"""  
import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
from WhaleTracking.whale_tracking import WhaleTracking
from Statistics.address_clustering import AddressClustering
from Helper.helperfunctions import address_to_scripthash
from WhaleTracking.whale_tracking import WhaleTracking
from NodeConnect.node_connect import NodeConnect
from node_data import RPC_USER_RASPI, RPC_PASSWORD_RASPI, RPC_HOST_RASPI, RPC_USER_LOCAL, RPC_PASSWORD_LOCAL, RPC_HOST_LOCAL




if __name__ == "__main__":
   
    node = NodeConnect(RPC_USER_LOCAL, RPC_PASSWORD_LOCAL, RPC_HOST_LOCAL).get_node()
    print("los gehts")
    whale_walett_tracking = WhaleTracking(node=node)
    clustering = AddressClustering()

    print("1")
    whale_addresses = whale_walett_tracking.get_whale_addresses()
    #clustered_addresses = clustering.run_clustering()
    #whale_addresses_merged = whale_walett_tracking.merge_with_clusters(whale_addresses, clustered_addresses)

    print(whale_addresses)

    #whale_addresses_merged = "bc1qkh4xr4x8hjra8wymcnyvzzxu6alxeerwkrufln"
    #test = address_to_scripthash(whale_addresses_merged)
    #print(test)
    
    #print("2")
    #whale_walett_tracking.track_whale_balances(whale_addresses_merged)

    #print("3")
    #for address in whale_addresses_merged:
    #    print(address)
    #    whale_walett_tracking.detect_whale_trends(address)

    print ("Ende")
   
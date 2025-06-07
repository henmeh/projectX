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
from node_data import RPC_USER_RASPI, RPC_PASSWORD_RASPI, RPC_HOST_RASPI




if __name__ == "__main__":
   
    node = NodeConnect(RPC_USER_RASPI, RPC_PASSWORD_RASPI, RPC_HOST_RASPI).get_node()
    print("los gehts")
    test = WhaleTracking(node, "/media/henning/Volume/Programming/projectX/src/test/mempool_test.db")
    #clustering = AddressClustering()

    address = "bc1q7cyrfmck2ffu2ud3rn5l5a8yv6f0chkp0zpemf"
    balance_satoshis = test.get_address_balance(address)
    
    if balance_satoshis is not None:
        print(f"Balance for {address}: {balance_satoshis} satoshis")
        print(f"≈ {balance_satoshis / 100000000:.8f} BTC")



    #print("1")
    #whale_addresses = whale_walett_tracking.get_whale_addresses()
    #clustered_addresses = clustering.run_clustering()
    #whale_addresses_merged = whale_walett_tracking.merge_with_clusters(whale_addresses, clustered_addresses)

    #print(whale_addresses)

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
   
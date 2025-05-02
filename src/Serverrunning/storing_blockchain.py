import sys
from time import sleep
sys.path.append('/media/henning/Volume/Programming/projectX/src/')

from BlockchainStoring.blockchain_storing import BlockchainStoring
from NodeConnect.node_connect import NodeConnect
from node_data import RPC_USER_RASPI, RPC_PASSWORD_RASPI, RPC_HOST_RASPI, RPC_USER_LOCAL, RPC_PASSWORD_LOCAL, RPC_HOST_LOCAL



if __name__ == "__main__":
    node = NodeConnect(RPC_USER_RASPI, RPC_PASSWORD_RASPI, RPC_HOST_RASPI).get_node()
    blockchain_storing = BlockchainStoring(node)

    latest_block_in_db = blockchain_storing.get_latest_stored_block()
    print(latest_block_in_db)

    blockchain_storing.delete_existing_block_data(latest_block_in_db)
    blockchain_storing.delete_existing_block_data(latest_block_in_db-1)
    blockchain_storing.delete_existing_block_data(latest_block_in_db-2)

    #latest_block_in_db = blockchain_storing.get_latest_stored_block()
    #print(latest_block_in_db)
    #latest_block = node.rpc_call("getblockcount", [])["result"]

    #if latest_block_in_db < latest_block:
    #    blockchain_storing.sync_blocks(latest_block_in_db + 1, latest_block)

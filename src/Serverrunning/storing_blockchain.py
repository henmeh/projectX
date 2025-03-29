import sys
from time import sleep
sys.path.append('/media/henning/Volume/Programming/projectX/src/')

from BlockchainStoring.blockchain_storing import BlockchainStoring
from NodeConnect.node_connect import NodeConnect
from node_data import RPC_USER_RASPI, RPC_PASSWORD_RASPI, RPC_HOST_RASPI, RPC_USER_LOCAL, RPC_PASSWORD_LOCAL, RPC_HOST_LOCAL



if __name__ == "__main__":
    node = NodeConnect(RPC_USER_RASPI, RPC_PASSWORD_RASPI, RPC_HOST_RASPI).get_node()
    blockchain_storing = BlockchainStoring(node)

    latest_block = node.rpc_call("getblockcount", [])["result"]
    blockchain_storing.sync_blocks(0, latest_block)

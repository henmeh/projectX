import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
from NodeConnect.node_connect import NodeConnect

class FakeNode(NodeConnect):
    """Mock node implementation for testing"""
    def __init__(self, test_data=None):
        self.test_data_mempool = test_data or [[255, 2205], [127, 3494], [63, 14443], [31, 45798], [15, 143348], [7, 807188], [3, 4436537], [1, 2274202], [0, 0]]
        self.test_data_wahle_tracking = test_data or {
            "transactions": {},
            "balances": {},
            "mempool_txids": []
        }


    def electrum_request(self, method):
        """Fake implementation for compatibility"""
        return {"result": self.test_data_mempool}
    

    def rpc_call(self, method, params):
        if method == "getrawtransaction":
            txid = params[0]
            return {
                "result": self.test_data_wahle_tracking["transactions"].get(txid, {
                    "txid": txid,
                    "size": 250,
                    "vsize": 200,
                    "weight": 800,
                    "vin": [{"txid": f"prev_{txid}", "vout": 0}],
                    "vout": [
                        {"value": 5.0, "scriptPubKey": {"address": "address1"}},
                        {"value": 3.0, "scriptPubKey": {"address": "address2"}}
                    ]
                })
            }
        elif method == "getrawmempool":
            return {"result": self.test_data_wahle_tracking["mempool_txids"]}
        elif method == "getaddressbalance":
            address = params[0]["addresses"][0]
            return {"result": {"balance": self.test_data_wahle_tracking["balances"].get(address, 1000) * 1e8}}
        return {"error": "Method not implemented"}


    def rpc_batch_call(self, method, params_list):
        return [self.rpc_call(method, params) for params in params_list]


    def add_test_transaction(self, txid, inputs, outputs, total_sent):
        """Add test transaction data"""
        self.test_data_wahle_tracking["transactions"][txid] = {
            "txid": txid,
            "size": 300,
            "vsize": 250,
            "weight": 1000,
            "vin": inputs,
            "vout": outputs,
            # Other fields as needed
        }
        # Add previous transaction for inputs
        for vin in inputs:
            prev_txid = vin["txid"]
            if prev_txid not in self.test_data_wahle_tracking["transactions"]:
                self.test_data_wahle_tracking["transactions"][prev_txid] = {
                    "txid": prev_txid,
                    "vout": [
                        {"value": vin["value"], "scriptPubKey": {"address": vin["address"]}}
                    ]
                }


    def set_balance(self, address, balance):
        """Set balance for an address"""
        self.test_data_wahle_tracking["balances"][address] = balance


    def set_mempool_txids(self, txids):
        """Set mempool transaction IDs"""
        self.test_data_wahle_tracking["mempool_txids"] = txids

class FailingNode(FakeNode):
            def electrum_request(self, method):
                raise ConnectionError("Simulated network error")
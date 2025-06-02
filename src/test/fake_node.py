import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
from NodeConnect.node_connect import NodeConnect

class FakeNode(NodeConnect):
    """Mock node implementation for testing"""
    def __init__(self, test_data=None):
        self.test_data = test_data or [[255, 2205], [127, 3494], [63, 14443], [31, 45798], [15, 143348], [7, 807188], [3, 4436537], [1, 2274202], [0, 0]]


    def electrum_request(self, method):
        """Fake implementation for compatibility"""
        return {"result": self.test_data}


class FailingNode(FakeNode):
            def electrum_request(self, method):
                raise ConnectionError("Simulated network error")
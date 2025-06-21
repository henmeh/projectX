import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
from WhaleTracking.whale_alert import AlertSender


class FakeAlertSender(AlertSender):
    """Mock alert sender for testing"""
    def __init__(self):
        self.sent_alerts = []
    
    def send_alert(self, message: str):
        """Store alerts instead of sending them"""
        self.sent_alerts.append(message)
    
    def clear(self):
        """Clear sent alerts"""
        self.sent_alerts = []
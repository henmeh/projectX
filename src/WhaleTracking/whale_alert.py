import datetime
import hashlib
import re
from Helper.helperfunctions import create_table, store_data, fetch_data_params

class AlertSender:
    """Base class for alert sending implementations"""
    def send_alert(self, message: str):
        raise NotImplementedError("Should be implemented in subclass")

class WhaleAlerts:
    ALERT_TYPES = {
        "WHALE_MOVE": 1000000,  # $1M USD
        "MINER_BRIBE": 5000000,  # $5M USD
        "EXCHANGE_FLOW": 2000000  # $2M USD
    }
    
    EXCHANGE_PATTERNS = [
        r"bc1q\w{38}",  # SegWit address pattern
        r"1\w{33}",     # Legacy address pattern
        r"3\w{33}",     # P2SH address pattern
    ]
    
    def __init__(self, db_path: str, alert_sender: AlertSender = None):
        self.db_path = db_path
        self.alert_sender = alert_sender
        
        # Create tables
        create_table(db_path, """CREATE TABLE IF NOT EXISTS alerted_events (
            event_hash TEXT PRIMARY KEY,
            timestamp DATETIME)""")
        
        create_table(db_path, """CREATE TABLE IF NOT EXISTS alert_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            txid TEXT,
            alert_type TEXT,
            amount_btc REAL,
            amount_usd REAL,
            fee_rate REAL)""")


    def get_alert_threshold(self, alert_type: str = "WHALE_MOVE") -> int:
        return self.ALERT_TYPES[alert_type]


    def _generate_event_hash(self, tx: dict) -> str:
        """Create unique hash for alert events"""
        hash_data = f"{tx['txid']}-{tx['sum_usd_sent']}"
        return hashlib.sha256(hash_data.encode()).hexdigest()


    def is_alerted(self, event_hash: str) -> bool:
        """Check if event has been alerted"""
        result = fetch_data_params(
            self.db_path,
            "SELECT 1 FROM alerted_events WHERE event_hash = ?",
            (event_hash,)
        )
        return bool(result)


    def detect_unusual_activity(self, tx: dict):
        """Detect and send alerts for unusual whale activity"""
        try:
            # Validate input
            if not isinstance(tx, dict):
                raise ValueError("Invalid transaction data")
                
            required_keys = {'txid', 'sum_btc_sent', 'sum_usd_sent', 
                            'fee_per_vbyte', 'tx_in_addr', 'tx_out_addr'}
            if not all(key in tx for key in required_keys):
                raise ValueError("Missing required transaction data")
            
            # Generate unique event hash
            event_hash = self._generate_event_hash(tx)
            
            # Skip if already alerted
            if self.is_alerted(event_hash):
                return
            
            # Determine alert type
            alert_type = "WHALE_MOVE"
            if tx["fee_per_vbyte"] > 1000:  # Extremely high fee
                alert_type = "MINER_BRIBE"
            elif self._is_exchange_flow(tx["tx_out_addr"]):
                alert_type = "EXCHANGE_FLOW"
            
            # Check if meets threshold
            if tx["sum_usd_sent"] < self.get_alert_threshold(alert_type):
                return
            
            # Send alert
            self._send_alert(tx, alert_type)
            self._mark_as_alerted(event_hash, tx, alert_type)
            
        except Exception as e:
            print(f"Error in detect_unusual_activity: {str(e)}")


    def _is_exchange_flow(self, addresses: list) -> bool:
        """Check if addresses match known exchange patterns"""
        for pattern in self.EXCHANGE_PATTERNS:
            if any(re.match(pattern, addr) for addr in addresses):
                return True
        return False


    def _send_alert(self, tx: dict, alert_type: str):
        """Send alert through the configured sender"""
        if self.alert_sender:
            message = self._format_alert(tx, alert_type)
            self.alert_sender.send_alert(message)


    def _format_alert(self, tx: dict, alert_type: str) -> str:
        """Create formatted alert message"""
        emoji = "🐳" if alert_type == "WHALE_MOVE" else "⛏️" if alert_type == "MINER_BRIBE" else "🏦"
        return (
            f"{emoji} *{alert_type.replace('_', ' ').title()} Alert!* {emoji}\n"
            f"💰 *{tx['sum_btc_sent']:.2f} BTC* (${tx['sum_usd_sent']:,.2f})\n"
            f"⛽ *Fee Rate:* {tx['fee_per_vbyte']:.1f} sat/vB\n"
            f"📥 *From:* {', '.join(tx['tx_in_addr'][:2])}...\n"
            f"📤 *To:* {', '.join(tx['tx_out_addr'][:2])}...\n"
            f"🕒 *Time:* {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n"
            f"🔗 [View Transaction](https://mempool.space/tx/{tx['txid']})"
        )


    def _mark_as_alerted(self, event_hash: str, tx: dict, alert_type: str):
        """Mark event as alerted and store alert history"""
        # Store in alerted events
        store_data(
            self.db_path,
            "INSERT OR IGNORE INTO alerted_events (event_hash, timestamp) VALUES (?, CURRENT_TIMESTAMP)",
            (event_hash,)
        )
        
        # Store in alert history
        store_data(
            self.db_path,
            """INSERT INTO alert_history 
            (txid, alert_type, amount_btc, amount_usd, fee_rate) 
            VALUES (?, ?, ?, ?, ?)""",
            (tx['txid'], alert_type, tx['sum_btc_sent'], tx['sum_usd_sent'], tx['fee_per_vbyte'])
        )